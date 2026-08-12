"""scripts/eval_code.py pure logic: mode resolution, resumable JSONL accounting with a
pinned config header, and pass@k summary math. Loaded via importlib since scripts/ isn't
a package; no model, no GPU, no network."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from microlab.evals.reference.metrics import pass_at_k

_SPEC = importlib.util.spec_from_file_location(
    "eval_code", Path(__file__).resolve().parents[2] / "scripts" / "eval_code.py")
ec = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ec)

HEADER = {"run": "runs/x", "dataset": "humaneval", "mode": "chat", "n": 1,
          "temperature": 0.0, "max_new": 400, "seed": 0, "limit": None}


def test_resolve_mode_passthrough(tmp_path):
    assert ec.resolve_mode("base", tmp_path) == "base"
    assert ec.resolve_mode("chat", tmp_path) == "chat"


def test_resolve_mode_auto_reads_serve_config(tmp_path):
    (tmp_path / "serve_config.json").write_text(json.dumps({"mode": "chat"}))
    assert ec.resolve_mode("auto", tmp_path) == "chat"


def test_resolve_mode_auto_without_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="pass --mode"):
        ec.resolve_mode("auto", tmp_path)


def test_read_resume_fresh_and_matching(tmp_path):
    out = tmp_path / "r.jsonl"
    assert ec.read_resume(out, HEADER) == set()
    lines = [{"_header": HEADER},
             {"task_id": "HumanEval/0", "sample": 0, "passed": True},
             {"task_id": "HumanEval/1", "sample": 0, "passed": False}]
    out.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    assert ec.read_resume(out, HEADER) == {"HumanEval/0#0", "HumanEval/1#0"}


def test_read_resume_header_mismatch_raises(tmp_path):
    out = tmp_path / "r.jsonl"
    out.write_text(json.dumps({"_header": {**HEADER, "max_new": 100}}) + "\n")
    with pytest.raises(ValueError, match="different config header"):
        ec.read_resume(out, HEADER)


def test_summarize_pass_at_1_greedy():
    recs = [{"task_id": f"T/{i}", "sample": 0, "passed": i < 3} for i in range(10)]
    s = ec.summarize(recs, n=1)
    assert s["n_tasks"] == 10
    assert s["pass@1"] == pytest.approx(0.3)
    assert "pass@10" not in s  # n=1 cannot estimate pass@10


def test_summarize_pass_at_k_matches_reference():
    # two tasks, n=10 samples: task A passes 3/10, task B passes 0/10
    recs = [{"task_id": "A", "sample": s, "passed": s < 3} for s in range(10)]
    recs += [{"task_id": "B", "sample": s, "passed": False} for s in range(10)]
    s = ec.summarize(recs, n=10)
    assert s["pass@1"] == pytest.approx((pass_at_k(10, 3, 1) + 0.0) / 2)
    assert s["pass@10"] == pytest.approx((pass_at_k(10, 3, 10) + 0.0) / 2)


def test_summarize_uneven_samples_raise():
    recs = [{"task_id": "A", "sample": 0, "passed": True},
            {"task_id": "A", "sample": 1, "passed": True},
            {"task_id": "B", "sample": 0, "passed": False}]
    with pytest.raises(ValueError, match="sample count"):
        ec.summarize(recs, n=2)


def test_missing_samples_partial_task_regenerates_without_duplication():
    done = {"Mbpp/11#0", "Mbpp/11#2"}
    assert ec.missing_samples("Mbpp/11", 3, done) == [1]
    assert ec.missing_samples("Mbpp/12", 3, done) == [0, 1, 2]
    done_all = {f"Mbpp/11#{s}" for s in range(3)}
    assert ec.missing_samples("Mbpp/11", 3, done_all) == []
