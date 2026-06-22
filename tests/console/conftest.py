from __future__ import annotations

import json
from pathlib import Path

import pytest

from microlab.console import auth
from microlab.console.app import create_app

PASSWORD = "test-password-123"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture(autouse=True)
def _reset_login_throttle():
    # The login rate-limiter is process-global; reset around every test so
    # one test's failed-login backoff can't lock out the next test.
    auth.reset_login_failures()
    yield
    auth.reset_login_failures()


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    _write_json(
        tmp_path / "site" / "content" / "phases.json",
        [
            {
                "id": "phase-0",
                "title": "Phase 0",
                "status": "current",
                "goal": "g",
                "tasks": [],
                "readingPaperIds": [],
            }
        ],
    )
    _write_json(tmp_path / "papers" / "manifest.json", [])
    (tmp_path / "site" / "content" / "synopses").mkdir(parents=True, exist_ok=True)
    dist = tmp_path / "site" / "dist"
    dist.mkdir(parents=True, exist_ok=True)
    (dist / "index.html").write_text("<!doctype html><title>Console</title>", encoding="utf-8")
    (dist / "public.html").write_text("<!doctype html><title>Public</title>", encoding="utf-8")
    (tmp_path / "plans").mkdir(exist_ok=True)
    (tmp_path / "plans" / "note.md").write_text("# Note\n\nbody\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def app(project_root: Path):
    application = create_app(project_root)
    application.config.update(TESTING=True)
    auth.set_password(application.instance_path, PASSWORD)
    return application


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def auth_client(app):
    c = app.test_client()
    with c.session_transaction() as sess:
        sess["authed"] = True
    return c
