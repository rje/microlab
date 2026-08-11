import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "eval_rerank", Path(__file__).resolve().parents[2] / "scripts" / "eval_rerank.py")
er = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(er)


def test_delivered_counts_any_passing_sample():
    rows = [{"_header": {}},
            {"task_id": "T1", "sample": 0, "passed": False},
            {"task_id": "T1", "sample": 1, "passed": True},
            {"task_id": "T2", "sample": 0, "passed": False},
            {"task_id": "T2", "sample": 1, "passed": False}]
    got = er.delivered(rows)
    assert got["n_tasks"] == 2 and got["k"] == 2
    assert got["delivered_correct"] == 1 and got["delivered_rate"] == 0.5
    assert got["pass@1_first_sample"] == 0.0


def test_delivered_empty_input_raises_error():
    rows = [{"_header": {}}]
    with pytest.raises(ValueError, match="no task rows"):
        er.delivered(rows)


def test_delivered_pass_first_sample_only():
    rows = [{"_header": {}},
            {"task_id": "T1", "sample": 0, "passed": True},
            {"task_id": "T1", "sample": 1, "passed": False},
            {"task_id": "T2", "sample": 0, "passed": False},
            {"task_id": "T2", "sample": 1, "passed": True}]
    got = er.delivered(rows)
    assert got["pass@1_first_sample"] == 0.5
