# Reading Workspace v1 — Per-Paper Notes + Progress

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the owner set a reading-state/depth and write notes per paper directly in the console, persisted server-side and surfaced on reload.

**Architecture:** A SQLite store (`microlab.db`, git-ignored) holds per-paper progress; note prose is written to `content/papers/<id>/notes.md`. New Flask endpoints expose progress + notes behind the existing session login, with CSRF on writes. `/api/state` is extended to include each paper's progress plus a CSRF token. The SPA's existing Reading panel gains a read-state/depth selector and an autosaving notes editor per paper card.

**Tech Stack:** Python 3.11, Flask, sqlite3 (stdlib), pytest + Flask test client; React + TypeScript + zod + vitest for the SPA. All Python via `/home/rje/anaconda3/bin/conda run -n microlab`.

---

## Scope

Plan 2 of the learning-console rollout (spec:
`docs/superpowers/specs/2026-06-19-microlab-learning-console-design.md`). It builds
the SQLite store and the notes/progress write path end-to-end. The AI overview +
per-section guide, the embedded PDF pane, and the `/paper-overview` skill are the
next increment.

## File Structure

- Create `src/microlab/console/store.py` — SQLite progress store (schema, upsert, query).
- Modify `src/microlab/console/content.py` — notes read/write helpers + `valid_paper_ids`.
- Modify `src/microlab/console/app.py` — extend `/api/state`; add progress + notes routes.
- Modify `site/src/state.ts` — schema (`paper.progress`, `csrfToken`) + mutation helpers.
- Modify `site/src/App.tsx` — `PaperCard` with read-state/depth selector + notes editor.
- Tests: `tests/console/test_store.py`, extend `tests/console/test_app.py`, extend
  `site/tests/state.test.ts` and `site/tests/App.test.tsx`.

Data locations (under project root): `microlab.db` (git-ignored via `*.db`),
`content/papers/<paperId>/notes.md` (git-tracked).

---

## Task 1: SQLite progress store

**Files:**
- Create: `src/microlab/console/store.py`
- Test: `tests/console/test_store.py`

- [ ] **Step 1: Write failing tests**

Create `tests/console/test_store.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from microlab.console import store


def test_get_all_progress_empty_when_no_db(tmp_path: Path):
    assert store.get_all_progress(tmp_path / "microlab.db") == {}


def test_upsert_and_read_progress(tmp_path: Path):
    db = tmp_path / "microlab.db"
    store.upsert_progress(db, "rope", "mapped", "implement")
    all_progress = store.get_all_progress(db)
    assert all_progress["rope"] == {"readState": "mapped", "depth": "implement"}


def test_upsert_overwrites(tmp_path: Path):
    db = tmp_path / "microlab.db"
    store.upsert_progress(db, "rope", "skimming", None)
    store.upsert_progress(db, "rope", "built", "implement")
    assert store.get_all_progress(db)["rope"] == {"readState": "built", "depth": "implement"}


def test_upsert_rejects_unknown_read_state(tmp_path: Path):
    with pytest.raises(ValueError, match="read_state"):
        store.upsert_progress(tmp_path / "microlab.db", "rope", "banana", None)


def test_upsert_rejects_unknown_depth(tmp_path: Path):
    with pytest.raises(ValueError, match="depth"):
        store.upsert_progress(tmp_path / "microlab.db", "rope", "mapped", "banana")
```

- [ ] **Step 2: Run to verify failure**

Run: `/home/rje/anaconda3/bin/conda run -n microlab pytest tests/console/test_store.py -v`
Expected: FAIL (`microlab.console.store` missing).

- [ ] **Step 3: Implement `store.py`**

Create `src/microlab/console/store.py`:

```python
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

READ_STATES = {"unread", "skimming", "mapped", "built", "mastered"}
DEPTHS = {"implement", "understand", "aware"}


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_progress (
                paper_id   TEXT PRIMARY KEY,
                read_state TEXT NOT NULL DEFAULT 'unread',
                depth      TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def get_all_progress(db_path: str | Path) -> dict[str, dict[str, str | None]]:
    if not Path(db_path).exists():
        return {}
    conn = _connect(db_path)
    try:
        rows = conn.execute("SELECT paper_id, read_state, depth FROM paper_progress").fetchall()
    finally:
        conn.close()
    return {
        row["paper_id"]: {"readState": row["read_state"], "depth": row["depth"]}
        for row in rows
    }


def upsert_progress(
    db_path: str | Path, paper_id: str, read_state: str, depth: str | None
) -> None:
    if read_state not in READ_STATES:
        raise ValueError(f"invalid read_state: {read_state!r}")
    if depth is not None and depth not in DEPTHS:
        raise ValueError(f"invalid depth: {depth!r}")
    init_db(db_path)
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO paper_progress (paper_id, read_state, depth, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET
                read_state = excluded.read_state,
                depth = excluded.depth,
                updated_at = excluded.updated_at
            """,
            (paper_id, read_state, depth, datetime.now(UTC).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()
```

- [ ] **Step 4: Run to verify pass**

Run: `/home/rje/anaconda3/bin/conda run -n microlab pytest tests/console/test_store.py -v`
Expected: 5 passed. Then `/home/rje/anaconda3/bin/conda run -n microlab ruff check src/microlab/console/store.py tests/console/test_store.py` — clean.

- [ ] **Step 5: Commit**

```bash
cd /home/rje/src/python/microlab
git add src/microlab/console/store.py tests/console/test_store.py
git commit -m "feat: sqlite per-paper progress store"
```

---

## Task 2: Notes helpers and paper-id validation in content.py

**Files:**
- Modify: `src/microlab/console/content.py`
- Test: `tests/console/test_content.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/console/test_content.py`:

```python
def test_valid_paper_ids_from_manifest(tmp_path: Path):
    write_json(
        tmp_path / "papers" / "manifest.json",
        [
            {
                "topic": "architecture",
                "title": "RoFormer",
                "authors": "Su",
                "year": 2021,
                "source_url": "https://arxiv.org/abs/2104.09864",
                "pdf_url": "https://arxiv.org/pdf/2104.09864",
                "filename": "roformer.pdf",
            }
        ],
    )
    assert content.valid_paper_ids(tmp_path) == {"roformer"}


def test_notes_round_trip(tmp_path: Path):
    assert content.read_notes(tmp_path, "roformer") == ""
    content.write_notes(tmp_path, "roformer", "# my notes\n\nRoPE rotates Q/K.\n")
    assert "RoPE rotates" in content.read_notes(tmp_path, "roformer")
    expected = tmp_path / "content" / "papers" / "roformer" / "notes.md"
    assert expected.is_file()
```

- [ ] **Step 2: Run to verify failure**

Run: `/home/rje/anaconda3/bin/conda run -n microlab pytest tests/console/test_content.py -k "valid_paper_ids or notes_round_trip" -v`
Expected: FAIL (functions undefined).

- [ ] **Step 3: Append helpers to `content.py`**

Append to `src/microlab/console/content.py`:

```python
def valid_paper_ids(project_root: Path) -> set[str]:
    return {paper["id"] for paper in load_papers(project_root)}


def notes_path(project_root: Path, paper_id: str) -> Path:
    return project_root / "content" / "papers" / paper_id / "notes.md"


def read_notes(project_root: Path, paper_id: str) -> str:
    path = notes_path(project_root, paper_id)
    return path.read_text(encoding="utf-8") if path.exists() else ""


def write_notes(project_root: Path, paper_id: str, body: str) -> None:
    path = notes_path(project_root, paper_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(path)
```

- [ ] **Step 4: Run to verify pass**

Run: `/home/rje/anaconda3/bin/conda run -n microlab pytest tests/console/test_content.py -v`
Expected: all pass (6 total). Ruff clean.

- [ ] **Step 5: Commit**

```bash
cd /home/rje/src/python/microlab
git add src/microlab/console/content.py tests/console/test_content.py
git commit -m "feat: notes file helpers and paper-id validation"
```

---

## Task 3: Progress + notes endpoints; extend /api/state

**Files:**
- Modify: `src/microlab/console/app.py`
- Test: `tests/console/test_app.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/console/test_app.py`:

```python
def _login(client):
    token = _csrf_from_login(client)
    client.post("/login", data={"password": "test-password-123", "csrf_token": token})
    return client.get("/api/state").get_json()["csrfToken"]


def test_state_includes_progress_and_csrf(auth_client):
    body = auth_client.get("/api/state").get_json()
    assert "csrfToken" in body and body["csrfToken"]
    # papers may be empty in the fixture project; just assert the key contract holds
    assert isinstance(body["papers"], list)


def test_progress_requires_csrf(auth_client):
    # authed but no csrf header
    resp = auth_client.post("/api/papers/mmlu/progress", json={"readState": "mapped"})
    assert resp.status_code == 403


def test_progress_round_trip(client, project_root):
    # add a paper to the fixture manifest so the id is valid
    import json

    (project_root / "papers" / "manifest.json").write_text(
        json.dumps(
            [
                {
                    "topic": "evaluation",
                    "title": "Measuring Massive Multitask Language Understanding",
                    "authors": "H",
                    "year": 2020,
                    "source_url": "https://arxiv.org/abs/2009.03300",
                    "pdf_url": "https://arxiv.org/pdf/2009.03300",
                    "filename": "mmlu.pdf",
                }
            ]
        ),
        encoding="utf-8",
    )
    csrf = _login(client)
    resp = client.post(
        "/api/papers/mmlu/progress",
        json={"readState": "mapped", "depth": "understand"},
        headers={"X-CSRF-Token": csrf},
    )
    assert resp.status_code == 200, resp.get_data(as_text=True)
    papers = client.get("/api/state").get_json()["papers"]
    mmlu = next(p for p in papers if p["id"] == "mmlu")
    assert mmlu["progress"] == {"readState": "mapped", "depth": "understand"}


def test_progress_rejects_unknown_paper(client):
    csrf = _login(client)
    resp = client.post(
        "/api/papers/not-a-paper/progress",
        json={"readState": "mapped"},
        headers={"X-CSRF-Token": csrf},
    )
    assert resp.status_code == 404


def test_progress_rejects_bad_read_state(client, project_root):
    import json

    (project_root / "papers" / "manifest.json").write_text(
        json.dumps(
            [
                {
                    "topic": "evaluation",
                    "title": "Measuring Massive Multitask Language Understanding",
                    "authors": "H",
                    "year": 2020,
                    "source_url": "https://arxiv.org/abs/2009.03300",
                    "pdf_url": "https://arxiv.org/pdf/2009.03300",
                    "filename": "mmlu.pdf",
                }
            ]
        ),
        encoding="utf-8",
    )
    csrf = _login(client)
    resp = client.post(
        "/api/papers/mmlu/progress",
        json={"readState": "banana"},
        headers={"X-CSRF-Token": csrf},
    )
    assert resp.status_code == 400


def test_notes_round_trip_endpoint(client, project_root):
    import json

    (project_root / "papers" / "manifest.json").write_text(
        json.dumps(
            [
                {
                    "topic": "evaluation",
                    "title": "Measuring Massive Multitask Language Understanding",
                    "authors": "H",
                    "year": 2020,
                    "source_url": "https://arxiv.org/abs/2009.03300",
                    "pdf_url": "https://arxiv.org/pdf/2009.03300",
                    "filename": "mmlu.pdf",
                }
            ]
        ),
        encoding="utf-8",
    )
    csrf = _login(client)
    assert client.get("/api/papers/mmlu/notes").get_json()["content"] == ""
    resp = client.post(
        "/api/papers/mmlu/notes",
        json={"content": "RoPE rotates Q/K."},
        headers={"X-CSRF-Token": csrf},
    )
    assert resp.status_code == 200
    assert "RoPE rotates" in client.get("/api/papers/mmlu/notes").get_json()["content"]
```

- [ ] **Step 2: Run to verify failure**

Run: `/home/rje/anaconda3/bin/conda run -n microlab pytest tests/console/test_app.py -k "progress or notes or csrf" -v`
Expected: new tests FAIL.

- [ ] **Step 3: Replace `register_content_routes` in `app.py`**

Replace the entire `register_content_routes` function with this version (adds `store`
import, extends `api_state`, adds progress + notes routes). Keep everything else in
`app.py` unchanged:

```python
def register_content_routes(app: Flask) -> None:
    from flask import jsonify, send_file

    from microlab.console import content, store

    root: Path = app.config["PROJECT_ROOT"]
    db_path = root / "microlab.db"

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

- [ ] **Step 4: Run to verify pass**

Run: `/home/rje/anaconda3/bin/conda run -n microlab pytest tests/console/ -v`
Expected: all pass. Then ruff clean on `src/microlab/console tests/console`.

- [ ] **Step 5: Add `microlab.db` to fixture cleanup note**

No code change — confirm `git status` does not show `microlab.db` (it's git-ignored via `*.db`). If it appears, stop and fix `.gitignore`.

- [ ] **Step 6: Commit**

```bash
cd /home/rje/src/python/microlab
git add src/microlab/console/app.py tests/console/test_app.py
git commit -m "feat: progress and notes endpoints with csrf, progress in /api/state"
```

---

## Task 4: SPA state schema + mutation helpers

**Files:**
- Modify: `site/src/state.ts`
- Test: `site/tests/state.test.ts`

- [ ] **Step 1: Add the failing schema test**

In `site/tests/state.test.ts`, add a test inside the existing `describe` block:

```typescript
  it("accepts paper progress and a csrf token", () => {
    const parsed = parseMicrolabState({
      phases: [],
      papers: [
        {
          id: "mmlu",
          topic: "evaluation",
          title: "MMLU",
          authors: "H",
          year: 2020,
          sourceUrl: "https://arxiv.org/abs/2009.03300",
          pdfUrl: "/papers/evaluation/mmlu.pdf",
          filename: "mmlu.pdf",
          progress: { readState: "mapped", depth: "understand" }
        }
      ],
      synopses: {},
      evalRuns: [],
      csrfToken: "tok"
    });
    expect(parsed.papers[0].progress?.readState).toBe("mapped");
    expect(parsed.csrfToken).toBe("tok");
  });
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/rje/src/python/microlab/site && npx vitest run tests/state.test.ts`
Expected: the new test FAILS (zod strips `progress`/`csrfToken`, so assertions are undefined).

- [ ] **Step 3: Extend the schema and add helpers in `state.ts`**

In `site/src/state.ts`, change `paperSchema` to add an optional `progress`, and add
`csrfToken` to `microlabStateSchema`:

```typescript
const paperProgressSchema = z.object({
  readState: z.string(),
  depth: z.string().nullable()
});

const paperSchema = z.object({
  id: z.string(),
  topic: z.string(),
  title: z.string(),
  authors: z.string(),
  year: z.number(),
  sourceUrl: z.string().url(),
  pdfUrl: z.string(),
  filename: z.string(),
  progress: paperProgressSchema.optional()
});
```

In `microlabStateSchema` add `csrfToken: z.string().optional()`. Add the exported
type and mutation helpers at the end of the file:

```typescript
export type PaperProgress = z.infer<typeof paperProgressSchema>;

const notesSchema = z.object({ paperId: z.string(), content: z.string() });

async function mutate(path: string, csrfToken: string, body: unknown): Promise<void> {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-CSRF-Token": csrfToken },
    body: JSON.stringify(body)
  });
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(detail.trim() || `Request failed: ${response.status}`);
  }
}

export function saveProgress(
  paperId: string,
  csrfToken: string,
  progress: { readState: string; depth: string | null }
): Promise<void> {
  return mutate(`/api/papers/${encodeURIComponent(paperId)}/progress`, csrfToken, progress);
}

export function saveNotes(paperId: string, csrfToken: string, content: string): Promise<void> {
  return mutate(`/api/papers/${encodeURIComponent(paperId)}/notes`, csrfToken, { content });
}

export async function fetchNotes(paperId: string): Promise<string> {
  const response = await fetch(`/api/papers/${encodeURIComponent(paperId)}/notes`);
  if (!response.ok) {
    throw new Error(`Failed to load notes: ${response.status}`);
  }
  return notesSchema.parse(await response.json()).content;
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /home/rje/src/python/microlab/site && npx vitest run tests/state.test.ts`
Expected: all state tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/rje/src/python/microlab
git add site/src/state.ts site/tests/state.test.ts
git commit -m "feat: spa state schema for progress + notes mutation helpers"
```

---

## Task 5: Reading panel — read-state selector + notes editor

**Files:**
- Modify: `site/src/App.tsx`
- Test: `site/tests/App.test.tsx`

- [ ] **Step 1: Add the failing test**

Append a test to `site/tests/App.test.tsx` (the fixture `state` there has one paper
`mmlu`; give it progress by adding `progress: { readState: "unread", depth: null }`
to that paper and `csrfToken: "tok"` to the state object first). Then add:

```typescript
  it("shows a read-state selector and saves on change", async () => {
    const fetchMock = vi.fn(async () => ({ ok: true, json: async () => ({}), text: async () => "" }));
    vi.stubGlobal("fetch", fetchMock);

    render(<App initialState={state} />);
    const select = screen.getByLabelText(/reading state for/i) as HTMLSelectElement;
    fireEvent.change(select, { target: { value: "mapped" } });

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/papers/mmlu/progress",
        expect.objectContaining({
          method: "POST",
          headers: expect.objectContaining({ "X-CSRF-Token": "tok" })
        })
      );
    });
  });
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/rje/src/python/microlab/site && npx vitest run tests/App.test.tsx`
Expected: FAIL (no read-state selector).

- [ ] **Step 3: Implement `PaperCard` and use it in `ReadingPanel`**

In `site/src/App.tsx`:

(a) Add imports at the top: extend the existing `./state` import to include
`PaperProgress`, `saveProgress`, `saveNotes`, `fetchNotes`. Add `useState` (already
imported) usage.

(b) Add these constants near the top of the file (after imports):

```typescript
const READ_STATES = ["unread", "skimming", "mapped", "built", "mastered"];
const DEPTHS = ["implement", "understand", "aware"];
```

(c) Replace the `papers.map(...)` body in `ReadingPanel` so each paper renders a
`<PaperCard>`, and pass `csrfToken` into `ReadingPanel`. Change `ReadingPanel`'s
props to `{ papers, synopses, csrfToken }` and update its single call site in `App`
(`<ReadingPanel papers={readingPapers} synopses={state.synopses} csrfToken={state.csrfToken ?? ""} />`).

(d) Add the `PaperCard` component:

```tsx
function PaperCard({
  paper,
  synopsis,
  csrfToken
}: {
  paper: Paper;
  synopsis: PaperSynopsis | undefined;
  csrfToken: string;
}) {
  const [readState, setReadState] = useState(paper.progress?.readState ?? "unread");
  const [depth, setDepth] = useState<string>(paper.progress?.depth ?? "");
  const [notesOpen, setNotesOpen] = useState(false);
  const [notes, setNotes] = useState("");
  const [notesLoaded, setNotesLoaded] = useState(false);
  const [saved, setSaved] = useState<string>("");

  const persist = (next: { readState?: string; depth?: string }) => {
    const rs = next.readState ?? readState;
    const d = next.depth ?? depth;
    saveProgress(paper.id, csrfToken, { readState: rs, depth: d === "" ? null : d })
      .then(() => setSaved("saved"))
      .catch((error: Error) => setSaved(error.message));
  };

  const openNotes = async () => {
    setNotesOpen((open) => !open);
    if (!notesLoaded) {
      try {
        setNotes(await fetchNotes(paper.id));
      } catch {
        setNotes("");
      }
      setNotesLoaded(true);
    }
  };

  useEffect(() => {
    if (!notesLoaded) {
      return;
    }
    const handle = setTimeout(() => {
      saveNotes(paper.id, csrfToken, notes)
        .then(() => setSaved("saved"))
        .catch((error: Error) => setSaved(error.message));
    }, 800);
    return () => clearTimeout(handle);
  }, [notes, notesLoaded, csrfToken, paper.id]);

  return (
    <article className="paper-card" key={paper.id}>
      <div className="paper-meta">
        <span>{paper.topic}</span>
        <span>{paper.year}</span>
      </div>
      <h3>{paper.title}</h3>
      <p className="authors">{paper.authors}</p>

      <div className="progress-controls">
        <label>
          <span className="visually-hidden">Reading state for {paper.title}</span>
          <select
            aria-label={`Reading state for ${paper.title}`}
            value={readState}
            onChange={(event) => {
              setReadState(event.target.value);
              persist({ readState: event.target.value });
            }}
          >
            {READ_STATES.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span className="visually-hidden">Depth for {paper.title}</span>
          <select
            aria-label={`Depth for ${paper.title}`}
            value={depth}
            onChange={(event) => {
              setDepth(event.target.value);
              persist({ depth: event.target.value });
            }}
          >
            <option value="">depth…</option>
            {DEPTHS.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
        {saved && <span className="save-state">{saved}</span>}
      </div>

      {synopsis && (
        <p className="synopsis-lede">{synopsis.oneSentence}</p>
      )}

      <div className="paper-actions">
        <button type="button" onClick={openNotes}>
          <FileText aria-hidden="true" />
          {notesOpen ? "Hide notes" : "Notes"}
        </button>
        <a href={paper.pdfUrl}>
          <FileText aria-hidden="true" />
          PDF
        </a>
        <a href={paper.sourceUrl} rel="noreferrer" target="_blank">
          <ExternalLink aria-hidden="true" />
          Source
        </a>
      </div>

      {notesOpen && (
        <textarea
          className="notes-editor"
          aria-label={`Notes for ${paper.title}`}
          value={notes}
          placeholder="Your notes (mechanism in your own words, what not to copy, questions)…"
          onChange={(event) => setNotes(event.target.value)}
          rows={8}
        />
      )}
    </article>
  );
}
```

Note: keep the richer `synopsis` study-notes block if you like, but the lede is
enough for this slice. Ensure `useEffect` is imported from `react` (it already is).

- [ ] **Step 4: Add minimal styles**

Append to `site/src/styles.css`:

```css
.progress-controls { display: flex; gap: 0.5rem; align-items: center; margin: 0.5rem 0; flex-wrap: wrap; }
.progress-controls select { background: #0d1117; color: inherit; border: 1px solid #30363d; border-radius: 6px; padding: 0.25rem 0.4rem; }
.save-state { font-size: 0.75rem; opacity: 0.7; }
.notes-editor { width: 100%; margin-top: 0.5rem; background: #0d1117; color: inherit; border: 1px solid #30363d; border-radius: 8px; padding: 0.6rem; font: inherit; resize: vertical; }
.visually-hidden { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); }
```

- [ ] **Step 5: Run SPA tests + build**

Run:
```bash
cd /home/rje/src/python/microlab/site
npx vitest run
npm run build
```
Expected: all vitest pass; build succeeds. If the existing App test that renders the
paper card breaks because the synopsis study blocks moved, update that test to assert
on the lede/title instead (do not delete coverage).

- [ ] **Step 6: Commit**

```bash
cd /home/rje/src/python/microlab
git add site/src/App.tsx site/src/styles.css site/tests/App.test.tsx
git commit -m "feat: per-paper reading-state selector and notes editor"
```

---

## Task 6: Full verification and deploy

**Files:** none new.

- [ ] **Step 1: Full backend suite + lint**

Run:
```bash
cd /home/rje/src/python/microlab
/home/rje/anaconda3/bin/conda run -n microlab pytest -q
/home/rje/anaconda3/bin/conda run -n microlab ruff check .
```
Expected: all pass, ruff clean.

- [ ] **Step 2: Confirm git-ignored runtime files are untracked**

Run: `git status --short`
Expected: no `microlab.db`. (Runtime `content/papers/*/notes.md` are intentionally
trackable, but none exist from tests because tests use tmp dirs.)

- [ ] **Step 3: Local end-to-end smoke (real DB write)**

```bash
cd /home/rje/src/python/microlab
/home/rje/anaconda3/envs/microlab/bin/python scripts/serve_site.py --port 8799 &
SRV=$!; sleep 3
JAR=$(mktemp)
CSRF=$(curl -s -c "$JAR" http://127.0.0.1:8799/login | grep -oP 'name="csrf_token" value="\K[^"]+')
curl -s -b "$JAR" -c "$JAR" -o /dev/null --data-urlencode "password=lets learn!" --data-urlencode "csrf_token=$CSRF" http://127.0.0.1:8799/login
APICSRF=$(curl -s -b "$JAR" http://127.0.0.1:8799/api/state | /home/rje/anaconda3/bin/conda run -n microlab python -c "import sys,json;print(json.load(sys.stdin)['csrfToken'])")
PID="attention-is-all-you-need"
echo "set progress: $(curl -s -b "$JAR" -o /dev/null -w '%{http_code}' -H "X-CSRF-Token: $APICSRF" -H 'Content-Type: application/json' -d '{"readState":"mapped","depth":"implement"}' http://127.0.0.1:8799/api/papers/$PID/progress) (expect 200)"
curl -s -b "$JAR" http://127.0.0.1:8799/api/state | /home/rje/anaconda3/bin/conda run -n microlab python -c "import sys,json;p=[x for x in json.load(sys.stdin)['papers'] if x['id']=='$PID'][0];print('progress now:',p['progress'])"
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; rm -f "$JAR" microlab.db
```
Expected: `set progress: 200` and `progress now: {'readState': 'mapped', 'depth': 'implement'}`. (Removes the scratch `microlab.db` afterward so the deploy starts clean.)

- [ ] **Step 4: Merge to main**

```bash
cd /home/rje/src/python/microlab
git branch -f main HEAD 2>/dev/null || true   # only if on a feature branch; otherwise already on main
git status --short
```
If work was done directly on `main`, skip the merge. Otherwise fast-forward `main` to
the work and `git checkout main`.

- [ ] **Step 5: Deploy**

```bash
cd /home/rje/src/python/microlab
cd site && npm run build && cd ..
systemctl --user restart microlab-site
sleep 4
systemctl --user is-active microlab-site
echo "/api/state (unauth): $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8765/api/state) (expect 401)"
```

- [ ] **Step 6: Verify live through nginx**

```bash
JAR=$(mktemp)
CSRF=$(curl -s -c "$JAR" https://microlab.rje.ai/login | grep -oP 'name="csrf_token" value="\K[^"]+')
curl -s -b "$JAR" -c "$JAR" -o /dev/null --data-urlencode "password=lets learn!" --data-urlencode "csrf_token=$CSRF" https://microlab.rje.ai/login
echo "authed state: $(curl -s -b "$JAR" -o /dev/null -w '%{http_code}' https://microlab.rje.ai/api/state) (expect 200)"
rm -f "$JAR"
```
Expected: 200.

## Acceptance Criteria

- `pytest -q` and `ruff check .` pass; `npx vitest run` and `npm run build` pass.
- `/api/state` returns each paper's `progress` and a `csrfToken`; progress + notes
  writes require auth + a valid `X-CSRF-Token`; unknown paper ids 404; bad read-state
  400.
- In the live console you can set a paper's read-state/depth and write notes, and
  they persist across reload.
- `microlab.db` is git-ignored; deployed and verified at `https://microlab.rje.ai`.

## Self-Review

- **Spec coverage:** Implements the SQLite progress store, the notes/progress write
  path with CSRF, and the `/api/state` progress surfacing from the spec's Data Model
  and Server sections. Overview/skill/PDF pane are explicitly deferred to the next plan.
- **Placeholder scan:** complete code in every step; no TBDs.
- **Type consistency:** `upsert_progress`, `get_all_progress`, `valid_paper_ids`,
  `read_notes`/`write_notes`, `saveProgress`/`saveNotes`/`fetchNotes`, and the
  `readState`/`depth` field names are used consistently across backend, schema, and SPA.
