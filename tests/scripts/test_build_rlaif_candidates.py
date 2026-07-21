"""scripts/build_rlaif_candidates.py: stop-truncation and the candidate filtering (empty drop,
exact-dup collapse, <2-distinct skip, overlong-prompt skip). The GPU sampling wrapper isn't
exercised here. Loaded via importlib since scripts/ isn't a package."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch

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


def test_sample_candidates_uses_one_batched_call(monkeypatch, tmp_path):
    tok = _tok(tmp_path)
    calls: list[tuple] = []

    def fake_generate(model, ids, max_new, temperature=0.0, generator=None):
        calls.append(tuple(ids.shape))
        # echo the prompt + one distinct new token per row (so decoded candidates differ)
        newtok = torch.arange(ids.size(0), dtype=torch.long).unsqueeze(1)
        return torch.cat([ids, newtok], dim=1)

    monkeypatch.setattr(bc, "generate_cached", fake_generate)
    out = bc.sample_candidates(object(), tok, "hi there", "cpu", k=4, temp=0.8, max_new=8,
                               base_seed=0)
    assert len(calls) == 1        # ONE batched call, not k sequential
    assert calls[0][0] == 4       # batch dimension == k
    assert len(out) == 4          # k candidates decoded


def test_write_jsonl_roundtrips(tmp_path):
    items = [{"instruction": "a", "prompt": "p", "candidates": ["x", "y"]}]
    out = tmp_path / "sub" / "cand.jsonl"
    assert bc.write_jsonl(items, out) == 1
    assert [json.loads(x) for x in out.read_text().splitlines()] == items


def test_build_candidates_writes_progressively_and_resumes(monkeypatch, tmp_path):
    # Progressive: each kept record is handed to the writer AS produced (crash-safety), tagged
    # with its source row index. Resume: start_row skips already-done rows entirely.
    tok = _tok(tmp_path)
    rows = [{"instruction": t, "context": "", "response": ""} for t in ("a", "b", "c")]
    scripted = iter([["x", "y"], ["p", "q"], ["m", "n"]])
    monkeypatch.setattr(bc, "sample_candidates",
                        lambda *a, **k: next(scripted))
    written = []
    items = bc.build_candidates(None, tok, rows, "cpu", 1024, k=2, writer=written.append)
    assert [w["row"] for w in written] == [0, 1, 2]
    assert written == items  # writer sees exactly the kept records, in order

    # resume from row 2: rows 0-1 untouched (scripted iterator provides only one draw)
    scripted = iter([["r", "s"]])
    written2 = []
    bc.build_candidates(None, tok, rows, "cpu", 1024, k=2, writer=written2.append, start_row=2)
    assert [w["row"] for w in written2] == [2]
    assert written2[0]["candidates"] == ["r", "s"]
