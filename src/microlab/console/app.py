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
    key_file.write_text(key, encoding="utf-8")
    key_file.chmod(0o600)
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
        if not next_path.startswith("/"):
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
    from flask import jsonify

    from microlab.console import content

    @app.route("/api/state")
    @auth.login_required
    def api_state():
        try:
            state = content.load_state(app.config["PROJECT_ROOT"])
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 500
        return jsonify(state)
