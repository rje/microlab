import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "eval_selection", Path(__file__).resolve().parents[2] / "scripts" / "eval_selection.py")
es = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(es)


def test_bank_by_task_orders_and_drops_header():
    rows = [{"_header": {}},
            {"task_id": "T", "sample": 1, "passed": False, "solution": "b"},
            {"task_id": "T", "sample": 0, "passed": True, "solution": "a"}]
    got = es.bank_by_task(rows)
    assert [r["sample"] for r in got["T"]] == [0, 1]
    with pytest.raises(ValueError):
        es.bank_by_task([{"_header": {}}])


def test_run_methods_selects_and_flags_oracle():
    cands = ["def f(): return 1", "def f(): return 2", "def f(): return 1"]
    passed = [False, True, False]
    sigs = [("ok", "1"), ("ok", "2"), ("ok", "1")]
    got = es.run_methods(cands, passed, sigs, None, seed=0)
    assert got["first"] == 0 and got["oracle"] == 1
    assert got["cluster_shortest"] in (0, 2)      # largest cluster = the wrong pair
    assert got["self_tests"] is None              # not provided -> None, not fabricated


def test_run_methods_raises_on_beat_the_oracle():
    # oracle sees no pass, but 'passed' claims the plurality pick passes -> leakage bug
    cands = ["a", "a"]
    with pytest.raises(RuntimeError):
        es.run_methods(cands, [True, False], None, None, oracle_override=[False, False])


def test_summarize_power_gate():
    per_task = {f"t{i}": {"first": False, "oracle": True} for i in range(10)}
    rep = es.summarize(per_task)
    assert rep["recoverable_gap"] == 10 and rep["underpowered"] is True
