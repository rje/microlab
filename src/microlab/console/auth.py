from __future__ import annotations

import functools
import hmac
import json
import os
import secrets
import sys
import time
from pathlib import Path

from flask import jsonify, redirect, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

AUTH_FILENAME = "auth.json"


def auth_file(instance_path: str | Path) -> Path:
    return Path(instance_path) / AUTH_FILENAME


def set_password(instance_path: str | Path, password: str) -> None:
    path = auth_file(instance_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps({"password_hash": generate_password_hash(password)}) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(data)


def load_password_hash(instance_path: str | Path) -> str | None:
    path = auth_file(instance_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("password_hash")


def verify_password(instance_path: str | Path, password: str) -> bool:
    stored = load_password_hash(instance_path)
    if not stored:
        return False
    return check_password_hash(stored, password)


# --- login rate-limiting (single-user, in-process) ---

_FAILED = {"count": 0, "next_allowed": 0.0}


def login_locked_seconds() -> float:
    return max(0.0, _FAILED["next_allowed"] - time.monotonic())


def record_login_failure() -> None:
    _FAILED["count"] += 1
    delay = min(30.0, float(2 ** min(_FAILED["count"], 5)))
    _FAILED["next_allowed"] = time.monotonic() + delay


def reset_login_failures() -> None:
    _FAILED["count"] = 0
    _FAILED["next_allowed"] = 0.0


# --- CSRF (synchronizer token in the session) ---


def ensure_csrf_token() -> str:
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return token


def csrf_ok(submitted: str | None) -> bool:
    expected = session.get("csrf_token", "")
    return bool(expected) and bool(submitted) and hmac.compare_digest(submitted, expected)


# --- login gate ---


def login_required(view):
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("authed"):
            if request.path.startswith("/api/"):
                return jsonify({"error": "authentication required"}), 401
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)

    return wrapped


def _main(argv: list[str] | None = None) -> int:
    import argparse
    import getpass

    parser = argparse.ArgumentParser(prog="python -m microlab.console.auth")
    sub = parser.add_subparsers(dest="command", required=True)
    sp = sub.add_parser("set-password", help="Set the console login password")
    sp.add_argument(
        "--instance-path",
        default=str(Path("instance")),
        help="Directory holding auth.json (default: ./instance)",
    )
    args = parser.parse_args(argv)

    if args.command == "set-password":
        pw = getpass.getpass("New console password: ")
        if len(pw) < 8:
            print("password too short (min 8 chars)", file=sys.stderr)
            return 1
        if pw != getpass.getpass("Confirm password: "):
            print("passwords do not match", file=sys.stderr)
            return 1
        set_password(args.instance_path, pw)
        print(f"password set at {auth_file(args.instance_path)}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
