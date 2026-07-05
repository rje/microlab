"""scripts/rlaif_judge.py pure logic: prompt building carries global indices + candidates, and
verdict parsing keeps only in-range picks for the right records. The codex subprocess isn't
exercised. Loaded via importlib since scripts/ isn't a package."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "rlaif_judge", Path(__file__).resolve().parents[2] / "scripts" / "rlaif_judge.py")
rj = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rj)


def _rec(instr, cands):
    return {"instruction": instr, "prompt": "p", "candidates": cands}


def test_build_prompt_lists_global_indices_and_candidates():
    batch = [(7, _rec("add 2+2", ["4", "five"])), (8, _rec("capital?", ["Paris", "London"]))]
    prompt = rj.build_judge_prompt(batch)
    assert "index 7: instruction='add 2+2'" in prompt
    assert "index 8: instruction='capital?'" in prompt
    assert "[0] '4'" in prompt and "[1] 'five'" in prompt
    assert "7, 8" in prompt  # the explicit index list to cover


def test_parse_verdicts_keeps_valid_drops_bad():
    batch = [(0, _rec("i", ["a", "b", "c"])), (1, _rec("j", ["x", "y"]))]
    text = json.dumps({"verdicts": [
        {"index": 0, "best": 0, "worst": 2},   # valid (3 candidates)
        {"index": 1, "best": 0, "worst": 5},   # worst out of range (2 candidates) -> drop
        {"index": 9, "best": 0, "worst": 1},   # index not in batch -> drop
        {"index": 1, "best": 1, "worst": 0},   # valid
    ]})
    got = rj.parse_verdicts(text, batch)
    assert got == [{"index": 0, "best": 0, "worst": 2}, {"index": 1, "best": 1, "worst": 0}]


def test_verdict_file_ok_resume(tmp_path):
    batch = [(i, _rec("i", ["a", "b"])) for i in range(10)]
    f = tmp_path / "v.json"
    assert not rj._verdict_file_ok(f, batch)               # missing
    f.write_text(json.dumps([{"index": i} for i in range(10)]))
    assert rj._verdict_file_ok(f, batch)                   # full coverage
    f.write_text(json.dumps([{"index": i} for i in range(5)]))
    assert not rj._verdict_file_ok(f, batch)               # only 50% -> not done
    f.write_text("not json")
    assert not rj._verdict_file_ok(f, batch)               # corrupt
