# Console Flask + Password-Auth Foundation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the Microlab Console from the hand-rolled `http.server` script to a Flask package and put the whole site behind password-only session login, with no change to what the dashboard shows.

**Architecture:** Move the existing pure content-loading functions verbatim into `src/microlab/console/content.py`, reimplement the request routing as Flask routes in `src/microlab/console/app.py`, and add `src/microlab/console/auth.py` for password hashing (Werkzeug), a CLI to set the password, a signed-cookie session login, login-form CSRF, and login rate-limiting. The app keeps binding to `127.0.0.1`; nginx still terminates TLS. This plan is behavior-preserving for reads plus a login gate — no new product features.

**Tech Stack:** Python 3.11, the `microlab` conda env, Flask (brings Werkzeug + itsdangerous), pytest with the Flask test client, ruff.

---

## Scope

This is Plan 1 of the learning-console rollout (see
`docs/superpowers/specs/2026-06-19-microlab-learning-console-design.md`). It delivers
the Flask migration and authentication only. Progress/notes/overview endpoints, the
reading-workspace SPA, and the content skills are Plan 2.

## File Structure

- `environment.yml` — add `flask`.
- `.gitignore` — ignore `instance/` (secrets) and `*.db`.
- `src/microlab/__init__.py` — package marker (new; `src/` does not exist yet).
- `src/microlab/console/__init__.py` — package marker.
- `src/microlab/console/content.py` — pure content loaders + path-safety helpers,
  moved verbatim from `scripts/serve_site.py`. One responsibility: read and validate
  project content from disk.
- `src/microlab/console/auth.py` — password storage/verify, `set-password` CLI,
  CSRF helpers, `login_required`, login rate-limit. One responsibility: authentication.
- `src/microlab/console/app.py` — Flask app factory and all routes. One
  responsibility: HTTP surface.
- `src/microlab/console/templates/login.html` — server-rendered login form.
- `scripts/serve_site.py` — becomes a thin entrypoint calling `create_app(...).run()`.
- `ops/systemd/microlab-site.service` — `ExecStart` runs the Flask entrypoint with
  `MICROLAB_HTTPS=1`.
- `tests/console/conftest.py` — fixtures: temp project root, app, client, login helper.
- `tests/console/test_content.py` — ported content/path-safety tests.
- `tests/console/test_auth.py` — password + CSRF + rate-limit unit tests.
- `tests/console/test_app.py` — Flask test-client route + auth-gate tests.
- `tests/test_site_server.py` — deleted (replaced by `tests/console/`).

---

## Task 1: Add Flask and the package skeleton

**Files:**
- Modify: `environment.yml`
- Modify: `.gitignore`
- Create: `src/microlab/__init__.py`
- Create: `src/microlab/console/__init__.py`

- [ ] **Step 1: Add Flask to `environment.yml`**

Add `flask` to the conda-forge dependency list (it is on conda-forge), after `requests`:

```yaml
  - requests
  - flask
  - tqdm
```

- [ ] **Step 2: Apply the environment update**

Run:

```bash
cd ~/src/python/microlab
/home/rje/anaconda3/bin/conda env update -n microlab -f environment.yml --prune
```

Expected: completes; `flask` installed.

- [ ] **Step 3: Verify Flask imports from the env**

Run:

```bash
/home/rje/anaconda3/bin/conda run -n microlab python -c "import flask, werkzeug; print(flask.__version__)"
```

Expected: a version string (Flask 3.x), no ImportError.

- [ ] **Step 4: Ignore secrets and databases**

Append to `.gitignore`:

```gitignore
instance/
*.db
```

- [ ] **Step 5: Create package markers**

Run:

```bash
cd ~/src/python/microlab
mkdir -p src/microlab/console/templates tests/console
touch src/microlab/__init__.py src/microlab/console/__init__.py
```

- [ ] **Step 6: Commit**

```bash
git add environment.yml .gitignore src/microlab/__init__.py src/microlab/console/__init__.py
git commit -m "chore: add flask and console package skeleton"
```

---

## Task 2: Move content loaders into `content.py`

The functions in `scripts/serve_site.py` that load and validate content are already
pure and parameterized by `project_root`/paths. Move them verbatim; only the HTTP
handler classes and `main()` are left behind (they are replaced by Flask in Task 5/6).

**Files:**
- Create: `src/microlab/console/content.py`
- Create: `tests/console/test_content.py`

- [ ] **Step 1: Write `content.py` with the moved functions**

Create `src/microlab/console/content.py` containing **verbatim copies** of these
names from `scripts/serve_site.py` (their bodies are unchanged):

- Constants: `PHASE_CONTENT`, `SYNOPSES_CONTENT`, `PAPER_MANIFEST`, `SITE_DIST`,
  `EVAL_RUNS`, `MARKDOWN_ALLOWED_DIRS`, `MARKDOWN_ALLOWED_ROOT_FILES`,
  `SPECIAL_PAPER_IDS`
- Functions: `read_json`, `slugify`, `paper_id_for`, `load_papers`, `load_synopses`,
  `artifact_url`, `load_eval_runs`, `validate_state`, `load_state`,
  `title_from_markdown`, `resolve_safe_path`, `resolve_markdown_path`,
  `load_markdown_document`, `resolve_artifact_path`

Start the file with this header (drop the `http.server`/`PROJECT_ROOT` bits — they
move to `app.py`):

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote

PHASE_CONTENT = Path("site/content/phases.json")
SYNOPSES_CONTENT = Path("site/content/synopses")
PAPER_MANIFEST = Path("papers/manifest.json")
SITE_DIST = Path("site/dist")
EVAL_RUNS = Path("runs/evals")
MARKDOWN_ALLOWED_DIRS = {"ops", "papers", "plans"}
MARKDOWN_ALLOWED_ROOT_FILES = {"AGENTS.md", "README.md"}
```

`load_state` keeps its signature `load_state(project_root: Path) -> dict[str, Any]`
and still calls `validate_state(state)` before returning (the loud cross-reference
guard added earlier). Do not add a `PROJECT_ROOT` default — callers always pass it.

- [ ] **Step 2: Write the ported content tests**

Create `tests/console/test_content.py` by adapting the three content-focused tests
from `tests/test_site_server.py` to import from the new module. Replace the
`load_server_module()` helper with a direct import:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from microlab.console import content


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_state_combines_phase_papers_synopses_and_eval_runs(tmp_path: Path):
    write_json(
        tmp_path / "site" / "content" / "phases.json",
        [
            {
                "id": "phase-0",
                "title": "Phase 0: Evaluation Harness",
                "status": "current",
                "goal": "Build a repeatable eval harness.",
                "tasks": [],
                "readingPaperIds": ["mmlu"],
            }
        ],
    )
    write_json(
        tmp_path / "site" / "content" / "synopses" / "phase-0.json",
        {"mmlu": {"paperId": "mmlu", "oneSentence": "x"}},
    )
    write_json(
        tmp_path / "papers" / "manifest.json",
        [
            {
                "topic": "evaluation",
                "title": "Measuring Massive Multitask Language Understanding",
                "authors": "Hendrycks et al.",
                "year": 2020,
                "source_url": "https://arxiv.org/abs/2009.03300",
                "pdf_url": "https://arxiv.org/pdf/2009.03300",
                "filename": "2020-hendrycks-mmlu.pdf",
            }
        ],
    )

    state = content.load_state(tmp_path)

    assert state["phases"][0]["id"] == "phase-0"
    assert state["papers"][0]["id"] == "mmlu"
    assert state["papers"][0]["pdfUrl"] == "/papers/evaluation/2020-hendrycks-mmlu.pdf"
    assert state["synopses"]["mmlu"]["paperId"] == "mmlu"


def test_validate_state_rejects_unknown_reading_id(tmp_path: Path):
    write_json(
        tmp_path / "site" / "content" / "phases.json",
        [
            {
                "id": "phase-0",
                "title": "P0",
                "status": "current",
                "goal": "g",
                "tasks": [],
                "readingPaperIds": ["does-not-exist"],
            }
        ],
    )
    write_json(tmp_path / "papers" / "manifest.json", [])

    with pytest.raises(ValueError, match="unknown paper id"):
        content.load_state(tmp_path)


def test_resolve_safe_path_rejects_traversal(tmp_path: Path):
    root = tmp_path / "papers"
    allowed = root / "evaluation" / "paper.pdf"
    allowed.parent.mkdir(parents=True)
    allowed.write_bytes(b"%PDF")

    assert content.resolve_safe_path(root, "evaluation/paper.pdf") == allowed
    with pytest.raises(ValueError, match="unsafe path"):
        content.resolve_safe_path(root, "../secrets.txt")


def test_load_markdown_document_reads_allowed_markdown(tmp_path: Path):
    md = tmp_path / "plans" / "environment-setup.md"
    md.parent.mkdir(parents=True)
    md.write_text("# Environment Setup\n\nUse the env.\n", encoding="utf-8")

    document = content.load_markdown_document(tmp_path, "plans/environment-setup.md")
    assert document["title"] == "Environment Setup"
    assert document["path"] == "plans/environment-setup.md"
```

- [ ] **Step 3: Run the content tests**

Run:

```bash
cd ~/src/python/microlab
/home/rje/anaconda3/bin/conda run -n microlab pytest tests/console/test_content.py -v
```

Expected: PASS (4 tests).

- [ ] **Step 4: Commit**

```bash
git add src/microlab/console/content.py tests/console/test_content.py
git commit -m "feat: move console content loaders into a package module"
```

---

## Task 3: Password storage and the set-password CLI

**Files:**
- Create: `src/microlab/console/auth.py`
- Create: `tests/console/test_auth.py`

- [ ] **Step 1: Write failing password tests**

Create `tests/console/test_auth.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run:

```bash
cd ~/src/python/microlab
/home/rje/anaconda3/bin/conda run -n microlab pytest tests/console/test_auth.py -v
```

Expected: FAIL (`microlab.console.auth` does not exist).

- [ ] **Step 3: Implement password storage + CLI in `auth.py`**

Create `src/microlab/console/auth.py`:

```python
from __future__ import annotations

import functools
import hmac
import json
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
```

- [ ] **Step 4: Run to verify pass**

Run:

```bash
cd ~/src/python/microlab
/home/rje/anaconda3/bin/conda run -n microlab pytest tests/console/test_auth.py -v
```

Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/microlab/console/auth.py tests/console/test_auth.py
git commit -m "feat: console password storage and set-password cli"
```

---

## Task 4: CSRF helpers, login gate, and rate-limiting

**Files:**
- Modify: `src/microlab/console/auth.py`
- Modify: `tests/console/test_auth.py`

- [ ] **Step 1: Add failing tests for the rate-limiter**

Append to `tests/console/test_auth.py`:

```python
def test_rate_limiter_backs_off_then_resets():
    auth.reset_login_failures()
    assert auth.login_locked_seconds() == 0.0
    auth.record_login_failure()
    auth.record_login_failure()
    assert auth.login_locked_seconds() > 0.0
    auth.reset_login_failures()
    assert auth.login_locked_seconds() == 0.0
```

- [ ] **Step 2: Run to verify failure**

Run:

```bash
cd ~/src/python/microlab
/home/rje/anaconda3/bin/conda run -n microlab pytest tests/console/test_auth.py::test_rate_limiter_backs_off_then_resets -v
```

Expected: FAIL (`login_locked_seconds` undefined).

- [ ] **Step 3: Add CSRF, gate, and rate-limit to `auth.py`**

Append to `src/microlab/console/auth.py`:

```python
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
```

- [ ] **Step 4: Run the auth tests**

Run:

```bash
cd ~/src/python/microlab
/home/rje/anaconda3/bin/conda run -n microlab pytest tests/console/test_auth.py -v
```

Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/microlab/console/auth.py tests/console/test_auth.py
git commit -m "feat: console csrf, login gate, and login rate-limiting"
```

---

## Task 5: Flask app factory, login/logout routes, and login page

**Files:**
- Create: `src/microlab/console/app.py`
- Create: `src/microlab/console/templates/login.html`
- Create: `tests/console/conftest.py`
- Create: `tests/console/test_app.py`

- [ ] **Step 1: Write the conftest fixtures**

Create `tests/console/conftest.py`:

```python
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
```

- [ ] **Step 2: Write failing app/auth-flow tests**

Create `tests/console/test_app.py`:

```python
from __future__ import annotations

import re


def _csrf_from_login(client) -> str:
    html = client.get("/login").get_data(as_text=True)
    match = re.search(r'name="csrf_token" value="([^"]+)"', html)
    assert match, "login form must include a csrf token"
    return match.group(1)


def test_api_state_requires_auth(client):
    response = client.get("/api/state")
    assert response.status_code == 401


def test_login_success_grants_state(client):
    token = _csrf_from_login(client)
    response = client.post(
        "/login",
        data={"password": "test-password-123", "csrf_token": token},
        follow_redirects=False,
    )
    assert response.status_code == 302
    state = client.get("/api/state")
    assert state.status_code == 200
    assert state.get_json()["phases"][0]["id"] == "phase-0"


def test_login_rejects_wrong_password(client):
    token = _csrf_from_login(client)
    response = client.post(
        "/login",
        data={"password": "nope", "csrf_token": token},
    )
    assert response.status_code == 401
    assert client.get("/api/state").status_code == 401


def test_login_rejects_missing_csrf(client):
    response = client.post("/login", data={"password": "test-password-123"})
    assert response.status_code == 400


def test_logout_clears_session(auth_client):
    assert auth_client.get("/api/state").status_code == 200
    assert auth_client.post("/logout").status_code in (200, 302)
    assert auth_client.get("/api/state").status_code == 401
```

- [ ] **Step 3: Run to verify failure**

Run:

```bash
cd ~/src/python/microlab
/home/rje/anaconda3/bin/conda run -n microlab pytest tests/console/test_app.py -v
```

Expected: FAIL (`microlab.console.app` does not exist).

- [ ] **Step 4: Write the login template**

Create `src/microlab/console/templates/login.html`:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Microlab Console — Sign in</title>
  <style>
    body { font-family: system-ui, sans-serif; background: #0f1419; color: #e6edf3;
           display: grid; place-items: center; min-height: 100vh; margin: 0; }
    form { background: #161b22; padding: 2rem; border-radius: 12px; width: 280px;
           box-shadow: 0 10px 40px rgba(0,0,0,.4); }
    h1 { font-size: 1.1rem; margin: 0 0 1rem; }
    input { width: 100%; padding: .6rem; margin: .3rem 0 1rem; border-radius: 6px;
            border: 1px solid #30363d; background: #0d1117; color: #e6edf3; box-sizing: border-box; }
    button { width: 100%; padding: .6rem; border: 0; border-radius: 6px;
             background: #2f81f7; color: white; font-weight: 600; cursor: pointer; }
    .error { color: #ff7b72; font-size: .85rem; margin: 0 0 1rem; }
  </style>
</head>
<body>
  <form method="post" action="{{ url_for('login') }}">
    <h1>Microlab Console</h1>
    {% if error %}<p class="error">{{ error }}</p>{% endif %}
    <input type="hidden" name="csrf_token" value="{{ csrf_token }}" />
    <input type="hidden" name="next" value="{{ next_path }}" />
    <label for="password">Password</label>
    <input id="password" name="password" type="password" autofocus autocomplete="current-password" />
    <button type="submit">Sign in</button>
  </form>
</body>
</html>
```

- [ ] **Step 5: Write `app.py` with the factory and auth routes**

Create `src/microlab/console/app.py`:

```python
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
    """Read routes are added in Task 6."""
```

- [ ] **Step 6: Add a minimal `/api/state` route so the auth tests can pass**

Add this inside `register_content_routes` in `app.py` (it is fleshed out in Task 6):

```python
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
```

- [ ] **Step 7: Run the app tests**

Run:

```bash
cd ~/src/python/microlab
/home/rje/anaconda3/bin/conda run -n microlab pytest tests/console/test_app.py -v
```

Expected: PASS (5 tests).

- [ ] **Step 8: Commit**

```bash
git add src/microlab/console/app.py src/microlab/console/templates/login.html tests/console/conftest.py tests/console/test_app.py
git commit -m "feat: flask app factory with password login and logout"
```

---

## Task 6: Port the remaining read routes behind the login gate

**Files:**
- Modify: `src/microlab/console/app.py`
- Modify: `tests/console/test_app.py`

- [ ] **Step 1: Add failing tests for the ported routes**

Append to `tests/console/test_app.py`:

```python
def test_markdown_route_requires_auth(client):
    assert client.get("/api/markdown?path=plans/note.md").status_code == 401


def test_markdown_route_returns_document_when_authed(auth_client):
    response = auth_client.get("/api/markdown?path=plans/note.md")
    assert response.status_code == 200
    assert response.get_json()["title"] == "Note"


def test_markdown_route_rejects_unpublished_path(auth_client):
    assert auth_client.get("/api/markdown?path=environment.yml").status_code == 400


def test_spa_index_requires_auth(client):
    response = client.get("/")
    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_spa_index_served_when_authed(auth_client):
    response = auth_client.get("/")
    assert response.status_code == 200
    assert b"Console" in response.data
```

- [ ] **Step 2: Run to verify failure**

Run:

```bash
cd ~/src/python/microlab
/home/rje/anaconda3/bin/conda run -n microlab pytest tests/console/test_app.py -k "markdown or spa" -v
```

Expected: FAIL (routes not defined → 404/405).

- [ ] **Step 3: Flesh out `register_content_routes`**

Replace the body of `register_content_routes` in `app.py` with the full set of read
routes (keep the `api_state` route from Task 5):

```python
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
```

Remove the now-duplicated `api_state` defined inside Task 5's stub (this replacement
is the single source).

- [ ] **Step 4: Run the full app test file**

Run:

```bash
cd ~/src/python/microlab
/home/rje/anaconda3/bin/conda run -n microlab pytest tests/console/test_app.py -v
```

Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add src/microlab/console/app.py tests/console/test_app.py
git commit -m "feat: port console read routes into flask behind login"
```

---

## Task 7: Entrypoint, systemd, cleanup, and full verification

**Files:**
- Modify: `scripts/serve_site.py`
- Modify: `ops/systemd/microlab-site.service`
- Delete: `tests/test_site_server.py`

- [ ] **Step 1: Replace `scripts/serve_site.py` with a thin entrypoint**

Overwrite `scripts/serve_site.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microlab.console.app import create_app  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Microlab Console.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args()
    app = create_app(args.project_root)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Set the password for local runs**

Run (you will be prompted twice; this writes `instance/auth.json`):

```bash
cd ~/src/python/microlab
/home/rje/anaconda3/bin/conda run -n microlab python -m microlab.console.auth set-password --instance-path instance
```

Expected: `password set at instance/auth.json`. Confirm it is ignored:

```bash
git check-ignore instance/auth.json
```

Expected: prints the path (ignored).

- [ ] **Step 3: Smoke-test the running server**

Run:

```bash
cd ~/src/python/microlab
PYTHONPATH=src /home/rje/anaconda3/bin/conda run -n microlab python scripts/serve_site.py --port 8799 &
sleep 2
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8799/api/state   # expect 401
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8799/login       # expect 200
kill %1
```

Expected: `401` then `200`.

- [ ] **Step 4: Update the systemd unit**

In `ops/systemd/microlab-site.service`, set the environment and keep the same command:

```ini
[Service]
Environment=MICROLAB_HTTPS=1
WorkingDirectory=/home/rje/src/python/microlab
ExecStart=/home/rje/anaconda3/bin/conda run -n microlab python scripts/serve_site.py --host 127.0.0.1 --port 8765
```

(Leave `[Unit]`, `Restart`, and `[Install]` unchanged.)

- [ ] **Step 5: Delete the obsolete server test**

The behavior is now covered by `tests/console/`. Run:

```bash
cd ~/src/python/microlab
git rm tests/test_site_server.py
```

- [ ] **Step 6: Full test + lint pass**

Run:

```bash
cd ~/src/python/microlab
/home/rje/anaconda3/bin/conda run -n microlab pytest -v
/home/rje/anaconda3/bin/conda run -n microlab ruff check .
```

Expected: all tests pass (eval-harness tests from earlier plans, `tests/test_phase_content.py`, and `tests/console/*`), ruff clean.

- [ ] **Step 7: Rebuild the SPA and confirm it still serves**

Run:

```bash
cd ~/src/python/microlab/site && npm run build
```

Expected: build succeeds (the SPA is unchanged; it now loads only after login).

- [ ] **Step 8: Commit**

```bash
cd ~/src/python/microlab
git add scripts/serve_site.py ops/systemd/microlab-site.service
git commit -m "feat: run console via flask entrypoint behind login"
```

---

## Acceptance Criteria

- `pytest -v` and `ruff check .` pass from the `microlab` env.
- `GET /api/state` returns 401 unauthenticated and the full state once logged in.
- The password is set only via `python -m microlab.console.auth set-password`; there
  is no HTTP route that sets or changes it.
- `instance/` (password hash + secret key) is git-ignored.
- The existing dashboard renders unchanged after login; no SPA behavior was altered.
- The systemd unit runs the Flask entrypoint with `MICROLAB_HTTPS=1`.

## Self-Review

- **Spec coverage:** Implements the spec's Authentication section (Flask, Werkzeug
  hashing, signed-cookie session, CSRF on the login form, rate-limit, CLI-only
  password, no web credential route) and the "Console Server → Flask package" port
  with `content.py`/`auth.py`/`app.py` boundaries. Progress/notes/overview endpoints,
  the SQLite store, the SPA workspace, and the `/paper-overview` skill are explicitly
  deferred to Plan 2.
- **Placeholder scan:** No TBDs; every code step shows complete code. The Task 5
  `api_state` stub is explicitly replaced by the full route in Task 6 (noted in
  both places).
- **Type/name consistency:** `set_password`, `verify_password`, `ensure_csrf_token`,
  `csrf_ok`, `login_required`, `login_locked_seconds`, `record_login_failure`,
  `reset_login_failures`, `create_app`, `register_content_routes`, and
  `content.load_state`/`resolve_safe_path`/`resolve_markdown_path`/
  `resolve_artifact_path`/`load_markdown_document` are used consistently across tasks.
