"""scripts/assemble_rlaif_prefs.py: verdict merging and best/worst -> pair mapping, including
the drops (no verdict, best==worst, out-of-range index, identical picked text). Loaded via
importlib since scripts/ isn't a package."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "assemble_rlaif_prefs",
    Path(__file__).resolve().parents[2] / "scripts" / "assemble_rlaif_prefs.py")
ar = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ar)


def _rec(cands):
    return {"instruction": "i", "prompt": "### Instruction:\ni\n\n### Response:\n",
            "candidates": cands}


def test_build_pairs_maps_best_and_worst():
    records = [_rec(["good", "meh", "bad"]), _rec(["x", "y"])]
    verdicts = {0: {"index": 0, "best": 0, "worst": 2}, 1: {"index": 1, "best": 1, "worst": 0}}
    pairs = ar.build_pairs(records, verdicts)
    assert pairs == [
        {"prompt": records[0]["prompt"], "chosen": "good", "rejected": "bad"},
        {"prompt": records[1]["prompt"], "chosen": "y", "rejected": "x"},
    ]


def test_build_pairs_drops_bad_verdicts():
    records = [_rec(["a", "b"]), _rec(["a", "b"]), _rec(["a", "b"]), _rec(["dup", "dup"])]
    verdicts = {
        0: {"best": 0, "worst": 0},   # best == worst -> drop
        1: {"best": 0, "worst": 9},   # worst out of range -> drop
        3: {"best": 0, "worst": 1},   # identical text after strip -> drop
        # record 2 has no verdict -> drop
    }
    assert ar.build_pairs(records, verdicts) == []


def test_load_verdicts_merges_files(tmp_path):
    (tmp_path / "b0.json").write_text(json.dumps([{"index": 0, "best": 1, "worst": 0}]))
    (tmp_path / "b1.json").write_text(json.dumps([{"index": 5, "best": 2, "worst": 1}]))
    merged = ar.load_verdicts(sorted(tmp_path.glob("*.json")))
    assert merged == {0: {"index": 0, "best": 1, "worst": 0},
                      5: {"index": 5, "best": 2, "worst": 1}}


def test_load_verdicts_skips_corrupt_file(tmp_path):
    good = tmp_path / "verdicts_00000.json"
    good.write_text(json.dumps([{"index": 0, "best": 1, "worst": 0}]))
    partial = tmp_path / "verdicts_00030.json"  # e.g. a kill mid-write
    partial.write_text('[{"index": 30, "best":')
    merged = ar.load_verdicts([good, partial])  # corrupt one skipped, not fatal
    assert merged == {0: {"index": 0, "best": 1, "worst": 0}}


def test_load_records_indexes_by_line(tmp_path):
    f = tmp_path / "c.jsonl"
    f.write_text("\n".join(json.dumps(_rec([str(i), str(i) + "b"])) for i in range(3)))
    recs = ar.load_records(f)
    assert len(recs) == 3 and recs[2]["candidates"] == ["2", "2b"]
