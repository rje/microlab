# Microlab Learning Console — Design Spec

- **Date:** 2026-06-19
- **Status:** Draft for review
- **Author:** rje + Claude
- **Supersedes:** the read-only synopsis model in `site/content/synopses/`

## Context

The Microlab Console today is a read-only dashboard: it shows phases, a reading
queue derived from `papers/manifest.json`, and pre-written paper synopses. The
owner is using the project to learn how SOTA labs build LLMs by reading ~46
papers alongside building the lab, under a real time constraint.

A static synopsis is priming, not learning. The console should become an active
**reading workspace** that supports a deliberate study method, captures the
owner's own understanding, and progressively surfaces more as they go deeper —
across all 14 curriculum phases.

## Goals

1. Support a time-efficient study method: multi-pass reading, depth triage, and
   active recall — not linear cover-to-cover reading.
2. Let the owner **take notes in the console while reading**, persisted reliably.
3. Provide **AI-generated overviews and per-section summaries**, with an easy way
   to **request progressively deeper explanations** on demand.
4. Track per-paper progress and surface it across the dashboard so the console
   "reveals more" as work advances.
5. Stay **reliable across all phases**: broken content or references fail loudly,
   never silently (consistent with the project's no-silent-fallback rule).
6. Authenticate the owner **in our own server** using the Flask ecosystem's vetted
   building blocks — no new auth service, no Basic Auth, no hand-rolled cryptography.

## Non-Goals

- Multi-user support. This is a single-user tool (`me@rjevans.net`).
- 2FA in the MVP. Password + session only; TOTP can be added later behind the same
  login flow if wanted.
- In-app LLM API access. Explanation generation runs through Claude Code skills,
  not a live model endpoint in the app (avoids key/secret/cost exposure).
- Replacing the eval harness or any training work. This is the learning surface only.

## The Study Method This Encodes

- **Multi-pass reading.** Pass 1 skim (abstract/intro/figures/conclusion) → Pass 2
  body (skip proofs) → Pass 3 implement (only for papers you build). The console
  models progress as a `readState`: `unread → skimming → mapped → built → mastered`.
- **Depth triage.** Each paper carries a `depth`: `implement` (read deeply + build),
  `understand` (Pass 2), or `aware` (Pass 1, know the one claim). Not all 46 papers
  deserve equal time; the console makes the triage explicit.
- **Active recall + spaced repetition.** Retention comes from answering questions
  from memory, not re-reading. The console stores recall cards and schedules
  reviews (SM-2).
- **Layered notes.** Seeds (AI) prime; the owner writes the mechanism-in-their-own-
  words, limitations/what-not-to-copy, and recall prompts. The seed layer and the
  owner layer never overwrite each other.

## Architecture Overview

Five components with clear boundaries:

1. **Content skills** (Claude Code skills) — procedures Claude runs to read a PDF
   and write structured study content to disk. The only LLM in the system.
2. **Content store (files)** — AI-authored documents per paper, git-tracked.
3. **State store (SQLite)** — interactive, queryable state: progress and recall.
4. **Console server** (`src/microlab/console/`, **Flask**) — reads content + state,
   serves the SPA, authenticates every request, and exposes CSRF-guarded write
   endpoints.
5. **Console SPA** (`site/`) — login screen, dashboard, and per-paper reading
   workspace.

**Auth/deployment:** a small Flask app authenticates every request via signed-cookie
sessions; TLS is terminated by the nginx the project already deploys; the app stays
bound to `127.0.0.1`. No new service.

### The console ↔ skill bridge (decision: file-based, "Approach A")

The browser can't read PDFs or generate explanations; Claude (in Claude Code) can.
So generation is decoupled via files:

```
Console shows a deep-dive is missing
   → displays the exact command:  /explain <paperId> "<section>"
   → owner runs it in Claude Code
   → Claude reads the PDF, writes content/papers/<id>/explained/<slug>.md
   → console (live file read) shows it on next load
```

No API keys, no in-app model, Claude-quality output, git-tracked. A future
enhancement (out of scope now) can auto-fulfill requests via a background loop
using the **same file contract** — so it is an addition, not a rewrite.

## Data Model

### Files (documents, git-tracked)

```
content/papers/<paperId>/
  overview.json          # AI: overview + per-section summaries  (skill-written)
  explained/
    <slug>.md            # AI: on-demand deep dives              (skill-written)
  notes.md               # owner: prose notes                    (console-written)
papers/<topic>/<file>.pdf  # existing PDFs (unchanged)
```

`<paperId>` is the id the server already derives from the manifest
(`slugify(title)`, with the five special-cased eval ids). No new id scheme.

**`overview.json`:**
```json
{
  "paperId": "roformer-enhanced-transformer-with-rotary-position-embedding",
  "generatedAt": "2026-06-19T00:00:00Z",
  "depthSuggestion": "implement",
  "tldr": "one sentence claim",
  "fullOverview": "markdown, a few paragraphs",
  "readingFocus": ["what to watch for", "..."],
  "sections": [
    { "id": "3-2", "title": "Rotary Position Embedding", "summary": "markdown" }
  ]
}
```
The server does **not** store a "has deep dive" flag in this file — it lists
`explained/` at read time so the flag can never go stale.

**`notes.md`:** prose only. Minimal descriptive frontmatter (`paperId`, `title`)
for the markdown renderer. Progress state lives in SQLite, not here, to avoid
duplicating state across two stores.

**`explained/<slug>.md`:** one deep dive, frontmatter records `paperId`,
`section`, and the `question` asked.

### SQLite (interactive state, git-ignored)

`microlab.db`, created and migrated by the app at startup. Sessions are **not**
stored here — Flask's signed-cookie session handles them.

```sql
CREATE TABLE paper_progress (
  paper_id     TEXT PRIMARY KEY,
  read_state   TEXT NOT NULL DEFAULT 'unread',
  depth        TEXT,                       -- implement | understand | aware
  last_section TEXT,
  updated_at   TEXT NOT NULL               -- ISO8601 UTC
);

CREATE TABLE recall_cards (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  paper_id      TEXT NOT NULL,
  question      TEXT NOT NULL,
  answer        TEXT,
  source        TEXT NOT NULL DEFAULT 'ai',  -- ai | me
  ease          REAL NOT NULL DEFAULT 2.5,
  interval_days INTEGER NOT NULL DEFAULT 0,
  due_at        TEXT NOT NULL,               -- ISO8601 date
  created_at    TEXT NOT NULL
);

CREATE TABLE recall_reviews (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  card_id     INTEGER NOT NULL REFERENCES recall_cards(id),
  reviewed_at TEXT NOT NULL,
  grade       INTEGER NOT NULL              -- 0..5 (SM-2)
);
```

`paper_id` is validated against the manifest-derived ids on every write; an
unknown id is rejected loudly (never silently inserted). Scheduling uses SM-2.

**Boundary rule:** documents (prose, AI explanations, PDFs) → files; queryable or
schedulable state (progress, recall) → SQLite. Data *shape* decides the store.

## Authentication (Flask, password-only)

Single-user, **signed-cookie session** auth built from the Flask ecosystem's vetted
parts. We write no cryptography of our own.

- **Password storage:** a single Werkzeug hash (`generate_password_hash`, scrypt)
  in a git-ignored file (`instance/auth.json`). No plaintext anywhere.
- **Password is settable ONLY via CLI:** `python -m microlab.console.auth set-password`,
  run from a shell on the host. There is **no web route** to set or change the
  password, so no authenticated session can alter credentials. Recovery = rerun the
  CLI on the box.
- **Login:** `GET /login` serves a form; `POST /login` verifies with
  `werkzeug.security.check_password_hash` (constant-time) and, on success, marks the
  Flask session authenticated. Failed attempts are rate-limited with incremental
  backoff.
- **Session:** Flask's `session` (signed by itsdangerous using a persisted
  `SECRET_KEY`) carries an authenticated flag and a CSRF token. The `SECRET_KEY` is
  generated once with `secrets.token_hex` and stored in the git-ignored
  `instance/` config — it is a signing key, never a password.
- **Cookie flags:** `HttpOnly`, `SameSite=Strict`, `Secure` when the request arrived
  over HTTPS (Flask `SESSION_COOKIE_*` config + `ProxyFix` reading nginx's
  `X-Forwarded-Proto`).
- **CSRF:** the session's CSRF token must be echoed in an `X-CSRF-Token` header on
  every write; compared with `hmac.compare_digest`. With `SameSite=Strict` this
  double-covers cross-site writes. (The SPA fetches the token from `/api/state`.)
- **Logout:** `POST /logout` clears the session.
- **Enforcement:** a `@login_required` decorator wraps all routes except `/login`
  and static assets; a `require_csrf` check wraps writes. The app binds `127.0.0.1`;
  nginx terminates TLS. Both reads and writes require a session.

**Provenance:** Werkzeug (hashing), itsdangerous (session signing), and stdlib
`secrets`/`hmac` — all bundled with Flask. No invented crypto.

## Console Server (`src/microlab/console/`, Flask)

The current `scripts/serve_site.py` (a hand-rolled `http.server`) is replaced by a
small Flask app in a testable package mirroring `src/microlab/evals/`:

```
src/microlab/console/
  app.py        # Flask app factory, routes/blueprints, static SPA + PDF serving
  content.py    # manifest, phases, overview, notes, explained loaders + validation
  store.py      # SQLite: schema, migrations, progress + recall queries
  auth.py       # password hash/verify (CLI), login_required, CSRF, rate-limit
scripts/serve_site.py   # thin entrypoint: create_app().run(host=127.0.0.1, port=8765)
```

- **New dependency:** `flask` in `environment.yml` (brings Werkzeug + itsdangerous).
- **Path safety:** Flask `send_from_directory` / Werkzeug `safe_join` replace the
  hand-rolled `resolve_safe_path` / traversal checks.
- The existing logic (state loading + validation, markdown rendering with its
  allow-list, PDF + artifact serving) ports into Flask routes; its tests port to the
  Flask test client.

### Endpoints

Auth: `GET /login`, `POST /login`, `POST /logout`.

Reads (require a valid session):
- `GET /api/state` — existing, extended to merge `paper_progress`, per-paper content
  availability (does `overview.json` exist? how many deep dives?), and the CSRF token.
- `GET /api/papers/<id>/overview`
- `GET /api/papers/<id>/notes`
- `GET /api/papers/<id>/explained/<slug>`
- `GET /api/recall/due` — cards due today (SQL).
- `GET /papers/...` — PDF serving.

Writes (require valid session **and** matching `X-CSRF-Token`):
- `POST /api/papers/<id>/notes` — write `notes.md`.
- `POST /api/papers/<id>/progress` — upsert `paper_progress`.
- `POST /api/recall/review` — insert review, reschedule via SM-2.
- `POST /api/recall/cards` — add an owner-authored card.

All writes validate `paper_id` against manifest ids and reject unknown ids,
malformed JSON, and missing fields loudly (HTTP 400/401/403 with a clear message the
SPA surfaces).

### Reliability

Extend `validate_state` to also check that any referenced `overview.json` parses and
its `paperId` matches, and that `explained/` slugs are well-formed. The
content-validation tests grow to cover the new content, the auth flows, and the
write endpoints (happy path + rejection cases).

## Console SPA (`site/`)

A login screen gates the app. Once authenticated, the dashboard stays; opening a
paper enters a **reading workspace** with three panes:

- **Read** — the PDF embedded (server already serves `/papers/...`).
- **Guide** — `tldr`, full overview, and collapsible per-section summaries; each
  section offers "go deeper": shows the loaded deep dive, or the `/explain` command
  to generate it.
- **Notes** — a live editor saving to `notes.md` (debounced autosave via
  `POST .../notes` with the CSRF header), a `readState` control, and a `depth` tag.
  Advancing `readState` updates progress, which flows back to the dashboard's
  per-phase reading bars.

A **Recall** view lists cards due today (`GET /api/recall/due`) as a self-quiz
(reveal answer, grade 0–5 → `POST /api/recall/review`).

State changes always round-trip through the server (no client-only state of record),
so a reload always reflects the truth on disk/DB.

## Component Interfaces (isolation)

- **Skills → content files:** one-way; skills only write files, never touch the DB
  or the running server.
- **Content files → server:** server reads + validates; never writes AI content.
- **SPA → server (writes):** the only mutation path for owner data; server owns the
  DB and `notes.md`; every write carries the session cookie + CSRF token.
- **`auth.py` → everything:** the single bounded place login, sessions, and CSRF are
  handled, unit-tested in isolation with the Flask test client.
- **nginx → app:** nginx only terminates TLS and forwards `X-Forwarded-Proto`; it
  performs no authentication. The app is the authentication boundary.

## Phasing (each phase = its own implementation plan)

1. **MVP** — the Flask `src/microlab/console/` package (porting current behavior);
   SQLite progress; password-only session auth (CLI-set password, login/logout,
   CSRF, rate-limit); `/paper-overview` skill; reading workspace (Read + Guide +
   Notes); `readState`/`depth`. Outcome: log in, read, get primed, take notes, track
   progress — securely.
2. **Deeper** — `/explain` skill; `explained/` rendering; "go deeper" affordances.
3. **Recall** — `/paper-recall` skill; recall tables + SM-2; the Recall quiz view.
4. **Optional later** — auto-fulfilled explanation requests (background loop) on the
   existing file contract; TOTP 2FA.

## Testing Strategy

- Python (Flask test client): content loaders + `validate_state` extensions;
  `store.py` against a temp DB (progress upsert, SM-2 scheduling, due queries);
  `auth.py` (password verify accept/reject, login rate-limit, CSRF accept/reject);
  endpoint tests including unauthenticated requests rejected with 401 and CSRF-less
  writes rejected with 403; ported markdown/PDF/path-safety tests.
- SPA: state parsing for the extended schema; login gate; workspace render; autosave
  sends the CSRF header; recall grading round-trips.
- Reliability: a real-content test that loads the project and fails on any broken
  reference (extends the existing `tests/test_phase_content.py`).

## Risks & Open Questions

- **Long PDFs** exceed a single Read; skills must read in page ranges and synthesize.
- **`SECRET_KEY` persistence:** generated once and stored in git-ignored `instance/`;
  rotating it logs the single user out (acceptable).
- **Session revocation:** signed-cookie sessions can't be revoked before expiry
  without a server-side store; acceptable for one user, and `Flask-Session` (SQLite
  backend) can be added later if revocation is ever needed.
- **`notes.md` autosave vs. external edits:** last-writer-wins; acceptable for a
  single user. Atomic write (temp + rename) avoids partial files.
- **Progress not in git:** SQLite is git-ignored; prose and AI explanations (the
  durable artifacts) are versioned; ephemeral state is not, by design.

## Self-Review

- Spec covers the owner's goals: in-console note-taking, AI overviews + per-section
  summaries + on-demand depth, progress surfacing, and in-app Flask password auth
  (not Basic Auth, no new service, password CLI-settable only) — across four build
  phases.
- No placeholders or TBDs.
- Storage boundary stated once and applied consistently (files vs SQLite).
- Auth uses only vetted, bundled libraries; no custom cryptography; credentials are
  not settable over the web.
- Scope is large; explicitly decomposed into four per-phase implementation plans,
  MVP first.
