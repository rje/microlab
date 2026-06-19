from __future__ import annotations

from pathlib import Path

from microlab.console import auth


def test_set_and_verify_password(tmp_path: Path):
    auth.set_password(tmp_path, "correct horse battery")
    assert auth.verify_password(tmp_path, "correct horse battery") is True
    assert auth.verify_password(tmp_path, "wrong") is False


def test_verify_password_false_when_unset(tmp_path: Path):
    assert auth.verify_password(tmp_path, "anything") is False


def test_auth_file_is_not_plaintext(tmp_path: Path):
    auth.set_password(tmp_path, "super-secret-value")
    stored = (tmp_path / "auth.json").read_text(encoding="utf-8")
    assert "super-secret-value" not in stored
