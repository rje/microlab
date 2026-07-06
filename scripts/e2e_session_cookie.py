"""Print a signed Flask session cookie authed for the console — for LOCAL authed e2e tests.

Reads the running app's secret key from ``instance/secret_key`` (same machine only) and signs a
``{"authed": true}`` session the way Flask would, so a Playwright browser can be authenticated
without the login password. Usage:

    MICROLAB_SESSION_COOKIE=$(python scripts/e2e_session_cookie.py) npx playwright test
"""

from __future__ import annotations

from pathlib import Path

from flask import Flask
from flask.sessions import SecureCookieSessionInterface


def session_cookie(instance_dir: str | Path = "instance") -> str:
    key = (Path(instance_dir) / "secret_key").read_text(encoding="utf-8").strip()
    app = Flask(__name__)
    app.secret_key = key
    return SecureCookieSessionInterface().get_signing_serializer(app).dumps({"authed": True})


if __name__ == "__main__":
    print(session_cookie())
