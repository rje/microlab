from __future__ import annotations

import json
import sys
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash

AUTH_FILENAME = "auth.json"


def auth_file(instance_path: str | Path) -> Path:
    return Path(instance_path) / AUTH_FILENAME


def set_password(instance_path: str | Path, password: str) -> None:
    path = auth_file(instance_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"password_hash": generate_password_hash(password)}) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


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
