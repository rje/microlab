"""scripts/eval_pairwise.py pure logic: prompt building, verdict parsing, and the
position-swap resolution (a model wins only if preferred in BOTH orderings). Generation and
codex calls aren't exercised. Loaded via importlib since scripts/ isn't a package."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "eval_pairwise", Path(__file__).resolve().parents[2] / "scripts" / "eval_pairwise.py")
ep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ep)


def test_resolve_pair_needs_agreement_across_orderings():
    # order AB: left=A. order BA: left=B.
    assert ep.resolve_pair("left", "right") == "A"   # A better both times (AB left, BA right)
    assert ep.resolve_pair("right", "left") == "B"   # B better both times
    assert ep.resolve_pair("left", "left") == "tie"  # AB says A, BA says B -> position bias
    assert ep.resolve_pair("tie", "right") == "tie"  # any tie -> tie
    assert ep.resolve_pair(None, "left") == "tie"    # missing verdict -> tie
    assert ep.resolve_pair("right", "right") == "tie"  # AB says B, BA says A -> disagree


def test_build_pair_prompt_labels_left_right_and_lists_ids():
    batch = [(0, "capital of France?", "Paris", "London"),
             (1, "capital of France?", "London", "Paris")]
    prompt = ep.build_pair_prompt(batch)
    assert "item 0: instruction='capital of France?'" in prompt
    assert "LEFT:  'Paris'" in prompt and "RIGHT: 'London'" in prompt
    assert "0, 1" in prompt


def test_parse_pair_verdicts_filters_unknown_and_bad():
    text = json.dumps({"verdicts": [
        {"item": 0, "better": "left"},
        {"item": 1, "better": "tie"},
        {"item": 9, "better": "left"},     # unknown id -> drop
        {"item": 2, "better": "maybe"},    # invalid label -> drop
    ]})
    got = ep.parse_pair_verdicts(text, {0, 1, 2})
    assert got == {0: "left", 1: "tie"}
