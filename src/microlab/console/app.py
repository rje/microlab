from __future__ import annotations

import os
import secrets
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

    from microlab.console import content

    root: Path = app.config["PROJECT_ROOT"]

    @app.route("/api/state")
    @auth.login_required
    def api_state():
        try:
            state = content.load_state(root)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 500
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
