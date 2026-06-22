from __future__ import annotations

import mimetypes
import os
import secrets
from datetime import date
from pathlib import Path

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


def create_app(project_root: str | Path | None = None) -> Flask:
    root = Path(project_root or PROJECT_ROOT).resolve()
    instance_path = root / "instance"
    app = Flask(__name__, instance_path=str(instance_path))
    app.config.update(
        PROJECT_ROOT=root,
        SECRET_KEY=_load_or_create_secret_key(instance_path),
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
