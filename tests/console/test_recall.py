from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from microlab.console import store


def test_review_state_empty_when_no_db(tmp_path: Path):
    assert store.get_review_state(tmp_path / "microlab.db") == {}


def test_first_good_review_schedules_one_day(tmp_path: Path):
    db = tmp_path / "microlab.db"
    res = store.record_review(db, "mmlu#1", "mmlu", 4, date(2026, 1, 1))
    assert res["intervalDays"] == 1
    assert res["dueAt"] == "2026-01-02"
    assert store.get_review_state(db)["mmlu#1"]["dueAt"] == "2026-01-02"


def test_second_good_review_schedules_six_days(tmp_path: Path):
    db = tmp_path / "microlab.db"
    store.record_review(db, "mmlu#1", "mmlu", 4, date(2026, 1, 1))
    res = store.record_review(db, "mmlu#1", "mmlu", 4, date(2026, 1, 2))
    assert res["intervalDays"] == 6
    assert res["dueAt"] == "2026-01-08"


def test_failing_review_resets_interval(tmp_path: Path):
    db = tmp_path / "microlab.db"
    store.record_review(db, "mmlu#1", "mmlu", 5, date(2026, 1, 1))
    store.record_review(db, "mmlu#1", "mmlu", 5, date(2026, 1, 2))
    res = store.record_review(db, "mmlu#1", "mmlu", 1, date(2026, 1, 8))
    assert res["intervalDays"] == 1
    assert res["reps"] == 0


def test_ease_decreases_on_hard_grade(tmp_path: Path):
    db = tmp_path / "microlab.db"
    res = store.record_review(db, "c1", "p", 3, date(2026, 1, 1))
    assert res["ease"] < 2.5  # grade 3 lowers ease
    assert res["ease"] >= 1.3


def test_grade_out_of_range_raises(tmp_path: Path):
    with pytest.raises(ValueError):
        store.record_review(tmp_path / "microlab.db", "c1", "p", 9, date(2026, 1, 1))


def test_reviews_are_logged(tmp_path: Path):
    db = tmp_path / "microlab.db"
    store.record_review(db, "c1", "p", 4, date(2026, 1, 1))
    store.record_review(db, "c1", "p", 5, date(2026, 1, 2))
    assert store.count_reviews(db) == 2
