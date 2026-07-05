"""scripts/build_sft_mix.py normalizers + JSONL writer. The HF-loading wrappers aren't
exercised here (no network in CI); the pure field-mapping is what can silently go wrong.
Loaded via importlib since scripts/ isn't a package."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "build_sft_mix", Path(__file__).resolve().parents[2] / "scripts" / "build_sft_mix.py")
bm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bm)


def test_normalize_alpaca_maps_fields_and_skips_empty():
    got = bm.normalize_alpaca({"instruction": "Sort these", "input": "3 1 2", "output": "1 2 3"})
    assert got == {"instruction": "Sort these", "context": "3 1 2", "response": "1 2 3"}
    # empty input -> context "" (still kept)
    assert bm.normalize_alpaca({"instruction": "Hi", "input": "", "output": "Hello"}) == \
        {"instruction": "Hi", "context": "", "response": "Hello"}
    # empty instruction or output -> dropped
    assert bm.normalize_alpaca({"instruction": "", "input": "x", "output": "y"}) is None
    assert bm.normalize_alpaca({"instruction": "q", "input": "x", "output": "  "}) is None


def test_normalize_no_robots_single_turn_only():
    single = {"messages": [{"role": "user", "content": "What is 2+2?"},
                           {"role": "assistant", "content": "4"}]}
    assert bm.normalize_no_robots(single) == \
        {"instruction": "What is 2+2?", "context": "", "response": "4"}
    # multi-turn -> dropped (we only take clean single-turn pairs)
    multi = {"messages": single["messages"] + [{"role": "user", "content": "and 3+3?"},
                                               {"role": "assistant", "content": "6"}]}
    assert bm.normalize_no_robots(multi) is None
    # wrong role order / empty content -> dropped
    assert bm.normalize_no_robots({"messages": [{"role": "assistant", "content": "hi"},
                                                {"role": "user", "content": "hi"}]}) is None
    assert bm.normalize_no_robots({"messages": [{"role": "user", "content": ""},
                                                {"role": "assistant", "content": "x"}]}) is None


def test_write_jsonl_roundtrips(tmp_path):
    rows = [{"instruction": "a", "context": "", "response": "b"},
            {"instruction": "c", "context": "d", "response": "e"}]
    out = tmp_path / "sub" / "mix.jsonl"
    assert bm.write_jsonl(rows, out) == 2
    assert [json.loads(x) for x in out.read_text().splitlines()] == rows
