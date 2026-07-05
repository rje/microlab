"""scripts/build_rlaif_candidates.py: stop-truncation and the candidate filtering (empty drop,
exact-dup collapse, <2-distinct skip, overlong-prompt skip). The GPU sampling wrapper isn't
exercised here. Loaded via importlib since scripts/ isn't a package."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from microlab.tokenizer.fast import FastTokenizer

_SPEC = importlib.util.spec_from_file_location(
    "build_rlaif_candidates",
    Path(__file__).resolve().parents[2] / "scripts" / "build_rlaif_candidates.py")
bc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bc)


def test_truncate_cuts_at_earliest_stop():
    assert bc.truncate("keep this\n### End\ndrop") == "keep this"
    assert bc.truncate("  spaced  ") == "spaced"


def _tok(tmp_path):
    return FastTokenizer.train(["hello world foo bar baz qux"] * 4, vocab_size=300,
                               save_path=str(tmp_path / "tok.json"))


def test_build_candidates_dedups_and_needs_two_distinct(monkeypatch, tmp_path):
    tok = _tok(tmp_path)
    rows = [
        {"instruction": "a", "context": "", "response": ""},  # 3 distinct -> kept
        {"instruction": "b", "context": "", "response": ""},  # all identical -> skip (<2 distinct)
        {"instruction": "c", "context": "", "response": ""},  # one empty, rest dup -> skip
    ]
    scripted = iter([
        ["x", "y", "x", "z"],       # -> dedup {x,y,z} = 3 distinct, kept
        ["same", "same", "same", "same"],   # -> 1 distinct, skipped
        ["", "  ", "only", "only"],         # -> empties dropped, {only} = 1 distinct, skipped
    ])
    monkeypatch.setattr(bc, "sample_candidates", lambda *a, **k: next(scripted))

    items = bc.build_candidates(None, tok, rows, "cpu", block_size=1024, k=4)
    assert len(items) == 1
    assert items[0]["candidates"] == ["x", "y", "z"]  # order-preserving dedup
    assert items[0]["instruction"] == "a" and "### Response:" in items[0]["prompt"]


def test_build_candidates_skips_overlong_prompt(monkeypatch, tmp_path):
    tok = _tok(tmp_path)
    called: list[int] = []
    monkeypatch.setattr(bc, "sample_candidates",
                        lambda *a, **k: called.append(1) or ["p", "q"])
    rows = [{"instruction": "a long enough instruction here", "context": "", "response": ""}]
    items = bc.build_candidates(None, tok, rows, "cpu", block_size=4, k=4, max_new=4)
    assert items == [] and called == []  # skipped before any sampling


def test_write_jsonl_roundtrips(tmp_path):
    items = [{"instruction": "a", "prompt": "p", "candidates": ["x", "y"]}]
    out = tmp_path / "sub" / "cand.jsonl"
    assert bc.write_jsonl(items, out) == 1
    assert [json.loads(x) for x in out.read_text().splitlines()] == items
