from __future__ import annotations

import mimetypes
import os
import secrets
from datetime import date
from pathlib import Path

import requests as _requests
from flask import (
    Flask,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.middleware.proxy_fix import ProxyFix

from microlab.console import auth

# Serve ES module workers (pdf.js) with a JS content type so browsers accept them.
mimetypes.add_type("text/javascript", ".mjs")

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_or_create_secret_key(instance_path: Path) -> str:
    key_file = instance_path / "secret_key"
    if key_file.exists():
        return key_file.read_text(encoding="utf-8").strip()
    instance_path.mkdir(parents=True, exist_ok=True)
    key = secrets.token_hex(32)
    fd = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(key)
    return key


def load_or_create_api_token(instance_path: Path) -> str:
    """Bearer token for programmatic clients (the eval harness) that can't do the login
    redirect. Same 0600 instance-dir pattern as the Flask secret key."""
    token_file = instance_path / "api_token"
    if token_file.exists():
        return token_file.read_text(encoding="utf-8").strip()
    instance_path.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(32)
    fd = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(token)
    return token


def create_app(project_root: str | Path | None = None) -> Flask:
    root = Path(project_root or PROJECT_ROOT).resolve()
    instance_path = root / "instance"
    app = Flask(__name__, instance_path=str(instance_path))
    app.config.update(
        PROJECT_ROOT=root,
        SECRET_KEY=_load_or_create_secret_key(instance_path),
        API_TOKEN=load_or_create_api_token(instance_path),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Strict",
        SESSION_COOKIE_SECURE=bool(int(os.environ.get("MICROLAB_HTTPS", "0"))),
    )
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

    @app.route("/login", methods=["GET", "POST"])
    def login():
        next_path = request.values.get("next", "/")
        if (
            not next_path.startswith("/")
            or next_path.startswith("//")
            or "\\" in next_path
            or not next_path.isprintable()
        ):
            next_path = "/"
        if request.method == "GET":
            return render_template(
                "login.html", csrf_token=auth.ensure_csrf_token(), next_path=next_path, error=None
            )
        if not auth.csrf_ok(request.form.get("csrf_token")):
            return (
                render_template(
                    "login.html",
                    csrf_token=auth.ensure_csrf_token(),
                    next_path=next_path,
                    error="Invalid form token. Try again.",
                ),
                400,
            )
        locked = auth.login_locked_seconds()
        if locked > 0:
            return (
                render_template(
                    "login.html",
                    csrf_token=auth.ensure_csrf_token(),
                    next_path=next_path,
                    error=f"Too many attempts. Wait {int(locked) + 1}s.",
                ),
                429,
            )
        if auth.verify_password(app.instance_path, request.form.get("password", "")):
            auth.reset_login_failures()
            session["authed"] = True
            return redirect(next_path)
        auth.record_login_failure()
        return (
            render_template(
                "login.html",
                csrf_token=auth.ensure_csrf_token(),
                next_path=next_path,
                error="Incorrect password.",
            ),
            401,
        )

    @app.route("/logout", methods=["POST"])
    def logout():
        session.clear()
        return redirect(url_for("login"))

    register_content_routes(app)
    return app


def register_content_routes(app: Flask) -> None:
    from flask import jsonify, send_file

    from microlab.console import content, store

    root: Path = app.config["PROJECT_ROOT"]
    db_path = root / "microlab.db"
    dist = root / content.SITE_DIST

    @app.route("/public")
    @app.route("/public/p/<path:rest>")
    def public_shell(rest: str = ""):
        return send_file(dist / "public.html")

    @app.route("/public/api/library")
    def public_library_api():
        return jsonify(content.public_library(root))

    @app.route("/public/api/papers/<paper_id>/cards")
    def public_cards(paper_id: str):
        return jsonify({"cards": content.read_cards(root, paper_id)})

    @app.route("/public/pdf/<paper_id>")
    def public_pdf(paper_id: str):
        path = content.public_pdf_path(root, paper_id)
        if path is None:
            return "", 404
        return send_file(path)

    @app.route("/assets/<path:filename>")
    def static_assets(filename: str):
        try:
            return send_file(content.resolve_safe_path(dist / "assets", filename))
        except (ValueError, FileNotFoundError):
            return "", 404

    @app.route("/api/state")
    @auth.login_required
    def api_state():
        try:
            state = content.load_state(root)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 500
        progress = store.get_all_progress(db_path)
        for paper in state["papers"]:
            paper["progress"] = progress.get(
                paper["id"], {"readState": "unread", "depth": None}
            )
        task_overrides = store.get_all_task_status(db_path)
        for phase in state["phases"]:
            overrides = task_overrides.get(phase["id"], {})
            for task in phase.get("tasks", []):
                if task["id"] in overrides:
                    task["status"] = overrides[task["id"]]
        state["csrfToken"] = auth.ensure_csrf_token()
        return jsonify(state)

    @app.route("/api/markdown")
    @auth.login_required
    def api_markdown():
        requested = request.args.get("path")
        if not requested:
            return jsonify({"error": "missing path"}), 400
        try:
            return jsonify(content.load_markdown_document(root, requested))
        except FileNotFoundError:
            return jsonify({"error": "not found"}), 404
        except ValueError:
            return jsonify({"error": "bad request"}), 400

    @app.route("/api/trajectory")
    @auth.login_required
    def api_trajectory():
        """Milestone tracking: the fixed-prompt completion sweeps plus milestone report
        paths, per run. Read from the working tree per request (like every content
        endpoint), so a new sweep or milestone doc appears without a redeploy."""
        import json as _json
        out = []
        for f in sorted((root / "evals" / "trajectory").glob("*-completions.jsonl")):
            run = f.name.removesuffix("-completions.jsonl")
            rows = [_json.loads(x) for x in f.read_text().splitlines() if x.strip()]
            steps = sorted({r["step"] for r in rows})
            comp: dict[str, dict[int, str]] = {}
            for r in rows:
                comp.setdefault(r["prompt_id"], {})[r["step"]] = r["completion"]
            docs = sorted(str(d.relative_to(root))
                          for d in (root / "docs").glob(f"{run}-*.md")
                          if not d.name.endswith("-prediction.md"))
            pred = root / "docs" / f"{run}-prediction.md"
            if pred.exists():
                docs.insert(0, str(pred.relative_to(root)))
            out.append({"run": run, "steps": steps, "completions": comp, "docs": docs})
        # Prompt text comes from the frozen prompt set so the UI can show what was asked.
        prompts = []
        try:
            import importlib.util as _ilu
            spec = _ilu.spec_from_file_location(
                "trajectory_prompts", root / "evals" / "trajectory_prompts.py")
            mod = _ilu.module_from_spec(spec)
            spec.loader.exec_module(mod)
            prompts = [{"id": q["id"], "text": q["text"]} for q in mod.PROMPTS]
        except FileNotFoundError:
            pass
        return jsonify({"runs": out, "prompts": prompts})

    @app.route("/api/papers/<paper_id>/progress", methods=["POST"])
    @auth.login_required
    def set_progress(paper_id: str):
        if not auth.csrf_ok(request.headers.get("X-CSRF-Token")):
            return jsonify({"error": "bad csrf token"}), 403
        if paper_id not in content.valid_paper_ids(root):
            return jsonify({"error": "unknown paper"}), 404
        data = request.get_json(silent=True) or {}
        try:
            store.upsert_progress(db_path, paper_id, data.get("readState"), data.get("depth"))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"ok": True})

    @app.route("/api/phases/<phase_id>/tasks/<task_id>/status", methods=["POST"])
    @auth.login_required
    def set_task_status_route(phase_id: str, task_id: str):
        if not auth.csrf_ok(request.headers.get("X-CSRF-Token")):
            return jsonify({"error": "bad csrf token"}), 403
        try:
            state = content.load_state(root)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 500
        phase = next((p for p in state["phases"] if p["id"] == phase_id), None)
        if phase is None:
            return jsonify({"error": "unknown phase"}), 404
        if not any(t["id"] == task_id for t in phase.get("tasks", [])):
            return jsonify({"error": "unknown task"}), 404
        data = request.get_json(silent=True) or {}
        try:
            store.set_task_status(db_path, phase_id, task_id, data.get("status"))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"ok": True})

    @app.route("/api/papers/<paper_id>/notes")
    @auth.login_required
    def get_notes(paper_id: str):
        if paper_id not in content.valid_paper_ids(root):
            return jsonify({"error": "unknown paper"}), 404
        return jsonify({"paperId": paper_id, "content": content.read_notes(root, paper_id)})

    @app.route("/api/papers/<paper_id>/notes", methods=["POST"])
    @auth.login_required
    def post_notes(paper_id: str):
        if not auth.csrf_ok(request.headers.get("X-CSRF-Token")):
            return jsonify({"error": "bad csrf token"}), 403
        if paper_id not in content.valid_paper_ids(root):
            return jsonify({"error": "unknown paper"}), 404
        data = request.get_json(silent=True) or {}
        content.write_notes(root, paper_id, str(data.get("content", "")))
        return jsonify({"ok": True})

    @app.route("/api/papers/<paper_id>/overview")
    @auth.login_required
    def get_overview(paper_id: str):
        if paper_id not in content.valid_paper_ids(root):
            return jsonify({"error": "unknown paper"}), 404
        data = content.read_overview(root, paper_id)
        if data is None:
            return jsonify({"error": "no overview"}), 404
        return jsonify(data)

    @app.route("/papers/<path:subpath>")
    @auth.login_required
    def papers(subpath: str):
        try:
            return send_file(content.resolve_safe_path(root / "papers", subpath))
        except (ValueError, FileNotFoundError):
            return "", 400

    @app.route("/artifacts/<path:subpath>")
    @auth.login_required
    def artifacts(subpath: str):
        try:
            return send_file(content.resolve_artifact_path(root, subpath))
        except (ValueError, FileNotFoundError):
            return "", 400

    @app.route("/api/recall/due")
    @auth.login_required
    def recall_due():
        cards = content.all_cards(root)
        state = store.get_review_state(db_path)
        today = date.today().isoformat()
        due = []
        for card in cards:
            st = state.get(card["id"])
            if st is None:
                due.append({**card, "status": "new"})
            elif str(st["dueAt"]) <= today:
                due.append({**card, "status": "due"})
        due.sort(key=lambda c: 0 if c["status"] == "new" else 1)
        return jsonify({"cards": due[:40], "total": len(due)})

    @app.route("/api/papers/<paper_id>/highlights")
    @auth.login_required
    def get_highlights(paper_id: str):
        if paper_id not in content.valid_paper_ids(root):
            return jsonify({"error": "unknown paper"}), 404
        return jsonify({"highlights": store.list_highlights(db_path, paper_id)})

    @app.route("/api/papers/<paper_id>/highlights", methods=["POST"])
    @auth.login_required
    def post_highlight(paper_id: str):
        if not auth.csrf_ok(request.headers.get("X-CSRF-Token")):
            return jsonify({"error": "bad csrf token"}), 403
        if paper_id not in content.valid_paper_ids(root):
            return jsonify({"error": "unknown paper"}), 404
        data = request.get_json(silent=True) or {}
        rects = data.get("rects")
        if not isinstance(rects, list):
            return jsonify({"error": "rects must be a list"}), 400
        hl = store.add_highlight(
            db_path, paper_id, int(data.get("page", 0)), rects, str(data.get("text", ""))
        )
        return jsonify(hl)

    @app.route("/api/papers/<paper_id>/highlights/<int:highlight_id>", methods=["DELETE"])
    @auth.login_required
    def delete_highlight_route(paper_id: str, highlight_id: int):
        if not auth.csrf_ok(request.headers.get("X-CSRF-Token")):
            return jsonify({"error": "bad csrf token"}), 403
        if paper_id not in content.valid_paper_ids(root):
            return jsonify({"error": "unknown paper"}), 404
        if not store.delete_highlight(db_path, paper_id, highlight_id):
            return jsonify({"error": "not found"}), 404
        return jsonify({"ok": True})

    @app.route("/api/recall/review", methods=["POST"])
    @auth.login_required
    def recall_review():
        if not auth.csrf_ok(request.headers.get("X-CSRF-Token")):
            return jsonify({"error": "bad csrf token"}), 403
        data = request.get_json(silent=True) or {}
        card_id = data.get("cardId")
        if card_id not in {c["id"] for c in content.all_cards(root)}:
            return jsonify({"error": "unknown card"}), 404
        try:
            result = store.record_review(
                db_path, card_id, str(data.get("paperId", "")), int(data.get("grade")), date.today()
            )
        except (ValueError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"ok": True, "next": result})

    def _api_auth_error():
        """Session auth OR bearer token (for the eval harness). Programmatic clients can't do
        the login redirect dance; the token lives in instance/api_token. Returns None when the
        request is authed, else a JSON ``(response, status)`` to return verbatim."""
        if session.get("authed"):
            return None
        header = request.headers.get("Authorization", "")
        if header.startswith("Bearer "):
            provided = header.removeprefix("Bearer ").strip()
            # compare_digest(str, str) raises TypeError on non-ASCII input, and the bearer
            # value comes straight from the client — compare bytes so a malformed token is
            # a 401, never a 500. (str.encode() can't raise for header-decoded text; the
            # except is belt-and-braces.)
            try:
                ok = secrets.compare_digest(provided.encode(), app.config["API_TOKEN"].encode())
            except UnicodeEncodeError:
                ok = False
            if ok:
                return None
            return jsonify({"error": "bad token"}), 401
        # Mirror auth.login_required's /api/* branch: 401 JSON, never a login redirect.
        return jsonify({"error": "authentication required"}), 401

    @app.route("/api/serve/runs")
    def api_serve_runs():
        auth_error = _api_auth_error()
        if auth_error is not None:
            return auth_error
        from microlab.console import serve
        return jsonify({"runs": serve.list_runs(root), "active": serve.active()})

    @app.route("/api/serve/reload", methods=["POST"])
    def api_serve_reload():
        # Force-reload a run to its latest checkpoint — how the owner picks up fresh
        # checkpoints mid-training. ``run`` omitted -> reload the currently-active run (or the
        # default if nothing is resident yet).
        auth_error = _api_auth_error()
        if auth_error is not None:
            return auth_error
        body = request.get_json(silent=True) or {}
        run = body.get("run")
        if run is not None and not isinstance(run, str):
            return jsonify({"error": "run must be a string"}), 400
        from microlab.console import serve
        if run is None:
            current = serve.active()
            run = current["name"] if current else None
        try:
            state = serve.get_state(root, run=run, reload=True)
        except FileNotFoundError as exc:
            return jsonify({"error": f"model not servable: {exc}"}), 503
        return jsonify({"run": state.run, "step": state.step})

    @app.route("/api/generate", methods=["POST"])
    def api_generate():
        auth_error = _api_auth_error()
        if auth_error is not None:
            return auth_error
        body = request.get_json(silent=True) or {}
        prompt = str(body.get("prompt", ""))
        if not prompt.strip():
            return jsonify({"error": "empty prompt"}), 400
        run = body.get("run")
        if run is not None and not isinstance(run, str):
            return jsonify({"error": "run must be a string"}), 400
        # Prior completed exchanges of the conversation, oldest first. Optional and backward
        # compatible: absent/empty means single-turn serving exactly as before. Shape is
        # validated here so the serving layer never sees a malformed conversation.
        history = body.get("history")
        if history is not None and (
            not isinstance(history, list)
            or not all(
                isinstance(t, dict)
                and isinstance(t.get("user"), str)
                and isinstance(t.get("assistant"), str)
                for t in history
            )
        ):
            return jsonify(
                {"error": "history must be a list of {user, assistant} string pairs"}), 400
        # Lazy import: keeps console restarts light (no torch at boot); the Playground pays
        # the import cost on first use.
        from microlab.console import serve
        try:
            state = serve.get_state(root, run=run)
        except FileNotFoundError as exc:
            return jsonify({"error": f"model not servable: {exc}"}), 503
        try:
            stream = serve.stream_generate(
                state, prompt,
                max_new_tokens=int(body.get("max_new_tokens", 128)),
                temperature=float(body.get("temperature", 0.8)),
                top_k=int(body["top_k"]) if body.get("top_k") else None,
                top_p=float(body["top_p"]) if body.get("top_p") else None,
                seed=int(body["seed"]) if body.get("seed") is not None else None,
                repetition_penalty=float(body.get("repetition_penalty", 1.0)),
                # Forces raw completion even on a chat model (skips template + stop strings) so
                # the Playground can offer a base-style side-by-side. No-op on base runs.
                raw=bool(body.get("raw", False)),
                history=history,
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        headers = {"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}
        # Turn accounting for the Playground's "older turns dropped" indicator. Only set when
        # the history actually shaped the prompt (chat run, raw off) — an ignored history on a
        # base run must not misreport as "all turns dropped".
        if stream.turns_used is not None:
            headers["X-Chat-Turns-Used"] = str(stream.turns_used)
            headers["X-Chat-Turns-Dropped"] = str(stream.turns_dropped)
        return app.response_class(stream, mimetype="text/plain", headers=headers)

    TENSORBOARD_UPSTREAM = os.environ.get("TENSORBOARD_URL", "http://127.0.0.1:6006")

    @app.route("/tensorboard/", defaults={"subpath": ""}, methods=["GET", "POST"])
    @app.route("/tensorboard/<path:subpath>", methods=["GET", "POST"])
    @auth.login_required
    def tensorboard_proxy(subpath: str):
        # Authed reverse-proxy to a local TensorBoard (127.0.0.1:6006, --path_prefix
        # /tensorboard). TensorBoard has no auth of its own, so login_required is the ONLY
        # thing standing between the public internet and it — do not remove.
        upstream = f"{TENSORBOARD_UPSTREAM}/tensorboard/{subpath}"
        fwd_headers = {"Accept": request.headers.get("Accept", "*/*")}
        if request.content_type:
            fwd_headers["Content-Type"] = request.content_type
        try:
            resp = _requests.request(
                method=request.method,
                url=upstream,
                params=request.args,
                data=request.get_data(),
                headers=fwd_headers,
                stream=True,
                timeout=30,
            )
        except _requests.exceptions.RequestException:
            return (
                "<h3>TensorBoard isn't running.</h3><p>Start it on the box: "
                "<code>tensorboard --logdir runs --path_prefix=/tensorboard "
                "--host 127.0.0.1 --port 6006</code></p>",
                503,
                {"Content-Type": "text/html"},
            )
        excluded = {"content-encoding", "content-length", "transfer-encoding", "connection"}
        headers = [(k, v) for k, v in resp.raw.headers.items() if k.lower() not in excluded]
        return app.response_class(
            resp.iter_content(chunk_size=8192), status=resp.status_code, headers=headers
        )

    @app.route("/")
    @app.route("/<path:requested>")
    @auth.login_required
    def spa(requested: str = ""):
        if requested.endswith(".md"):
            try:
                return send_file(content.resolve_markdown_path(root, requested))
            except FileNotFoundError:
                return "", 404
            except ValueError:
                return "", 400
        dist = root / content.SITE_DIST
        if requested:
            try:
                candidate = content.resolve_safe_path(dist, requested)
            except ValueError:
                return "", 400
            if candidate.is_file():
                return send_file(candidate)
        return send_file(dist / "index.html")
