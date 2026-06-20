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
