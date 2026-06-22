from __future__ import annotations

from pathlib import Path

import pytest

from microlab.console import store


def test_get_all_progress_empty_when_no_db(tmp_path: Path):
    assert store.get_all_progress(tmp_path / "microlab.db") == {}


def test_read_paths_tolerate_db_created_by_older_schema(tmp_path: Path):
    # Regression: a microlab.db created by an older schema has only the tables
    # that existed then. Read paths that SELECT from a newer table must not 500
    # (init_db is idempotent and back-fills the missing tables).
    import sqlite3

    db = tmp_path / "microlab.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE paper_progress "
        "(paper_id TEXT PRIMARY KEY, read_state TEXT, depth TEXT, updated_at TEXT)"
    )
    conn.commit()
    conn.close()

    assert store.get_all_task_status(db) == {}
    assert store.get_review_state(db) == {}
    assert store.count_reviews(db) == 0
    assert store.get_all_progress(db) == {}


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


def test_task_status_empty_when_no_db(tmp_path: Path):
    assert store.get_all_task_status(tmp_path / "microlab.db") == {}


def test_set_and_get_task_status(tmp_path: Path):
    db = tmp_path / "microlab.db"
    store.set_task_status(db, "phase-0", "eval-schema", "done")
    assert store.get_all_task_status(db) == {"phase-0": {"eval-schema": "done"}}


def test_set_task_status_overwrites(tmp_path: Path):
    db = tmp_path / "microlab.db"
    store.set_task_status(db, "phase-0", "eval-schema", "active")
    store.set_task_status(db, "phase-0", "eval-schema", "done")
    assert store.get_all_task_status(db)["phase-0"]["eval-schema"] == "done"


def test_set_task_status_rejects_unknown_status(tmp_path: Path):
    with pytest.raises(ValueError, match="status"):
        store.set_task_status(tmp_path / "microlab.db", "phase-0", "t1", "banana")
