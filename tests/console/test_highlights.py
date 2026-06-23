from __future__ import annotations

from pathlib import Path

from microlab.console import store


def test_list_highlights_empty(tmp_path: Path):
    assert store.list_highlights(tmp_path / "microlab.db", "mmlu") == []


def test_add_and_list_highlight(tmp_path: Path):
    db = tmp_path / "microlab.db"
    rects = [{"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.02}]
    hl = store.add_highlight(db, "mmlu", 1, rects, "some quoted text")
    assert hl["id"] >= 1
    assert hl["page"] == 1
    assert hl["rects"] == rects
    assert hl["text"] == "some quoted text"
    rows = store.list_highlights(db, "mmlu")
    assert len(rows) == 1
    assert rows[0]["rects"] == rects
    assert rows[0]["paperId"] == "mmlu"


def test_highlights_are_scoped_per_paper(tmp_path: Path):
    db = tmp_path / "microlab.db"
    store.add_highlight(db, "mmlu", 1, [{"x": 0, "y": 0, "w": 0.1, "h": 0.01}], "a")
    store.add_highlight(db, "helm", 2, [{"x": 0, "y": 0, "w": 0.1, "h": 0.01}], "b")
    assert len(store.list_highlights(db, "mmlu")) == 1
    assert len(store.list_highlights(db, "helm")) == 1


def test_delete_highlight(tmp_path: Path):
    db = tmp_path / "microlab.db"
    hl = store.add_highlight(db, "mmlu", 1, [{"x": 0, "y": 0, "w": 0.1, "h": 0.01}], "a")
    assert store.delete_highlight(db, "mmlu", hl["id"]) is True
    assert store.list_highlights(db, "mmlu") == []
    assert store.delete_highlight(db, "mmlu", hl["id"]) is False  # already gone


def test_read_path_tolerates_old_schema(tmp_path: Path):
    # mirrors the existing regression guard: a DB created before this table existed
    import sqlite3

    db = tmp_path / "microlab.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE paper_progress"
        " (paper_id TEXT PRIMARY KEY, read_state TEXT, depth TEXT, updated_at TEXT)"
    )
    conn.commit()
    conn.close()
    assert store.list_highlights(db, "mmlu") == []
