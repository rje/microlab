"""scripts/eval_passkey.py pure logic: exact-token-length prompt construction with the key
buried at a controlled depth, deterministic key drawing, answer extraction/scoring, and the
cell runner (exercised against a stubbed generator — no GPU). Loaded via importlib since
scripts/ isn't a package."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from microlab.tokenizer.fast import FastTokenizer

_SPEC = importlib.util.spec_from_file_location(
    "eval_passkey", Path(__file__).resolve().parents[2] / "scripts" / "eval_passkey.py")
ep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ep)


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    corpus = [
        ep.PREAMBLE, ep.FILLER * 8, ep.QUERY,
        "The pass key is 12345. Remember it. 12345 is the pass key. ",
        "0 1 2 3 4 5 6 7 8 9 01234 56789 98765 43210",
    ]
    return FastTokenizer.train(corpus * 4, vocab_size=400)


def _find_sub(haystack: list[int], needle: list[int]) -> int:
    """Index of the first occurrence of `needle` as a contiguous subsequence, else -1."""
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return -1


def test_prompt_is_exactly_length_tokens(tok):
    for length in (128, 256, 517):
        for depth in (0.1, 0.5, 0.9):
            ids = ep.build_passkey_prompt(tok, length, depth, "48213")
            assert len(ids) == length, (length, depth)


def test_prompt_contains_key_and_ends_with_query(tok):
    ids = ep.build_passkey_prompt(tok, 256, 0.5, "48213")
    text = tok.decode(ids)
    assert "The pass key is 48213" in text
    assert text.endswith(ep.QUERY)
    # the query tokens are the literal tail of the prompt (nothing appended after them)
    query_ids = tok.encode(ep.QUERY)
    assert ids[-len(query_ids):] == query_ids


def test_prompt_depth_controls_key_position(tok):
    key_ids = tok.encode(ep.KEY_SENTENCE.format(key="48213"))
    positions = []
    for depth in (0.1, 0.5, 0.9):
        ids = ep.build_passkey_prompt(tok, 512, depth, "48213")
        at = _find_sub(ids, key_ids)
        assert at != -1, f"key sentence not found at depth {depth}"
        positions.append(at)
    assert positions[0] < positions[1] < positions[2]
    # roughly proportional: depth 0.9 sits in the last third, 0.1 in the first third
    assert positions[0] < 512 * 0.34
    assert positions[2] > 512 * 0.66


def test_prompt_too_short_raises(tok):
    with pytest.raises(ValueError, match="too short"):
        ep.build_passkey_prompt(tok, 8, 0.5, "48213")


def test_extract_key():
    assert ep.extract_key(" 48213.") == "48213"
    assert ep.extract_key("\nThe pass key is 90210, remember") == "90210"
    assert ep.extract_key(" I do not know") is None
    assert ep.extract_key("") is None


def test_draw_key_is_deterministic_five_digits():
    a = ep.draw_key(0, 1024, 0.5, 3)
    assert a == ep.draw_key(0, 1024, 0.5, 3)  # same cell/sample -> same key
    assert len(a) == 5 and a.isdigit() and a[0] != "0"
    # different sample / cell / seed -> (overwhelmingly) different keys
    assert a != ep.draw_key(0, 1024, 0.5, 4)
    assert a != ep.draw_key(1, 1024, 0.5, 3)


def _echo_generate(tok):
    """A stub generate_cached: reads the key out of the decoded prompt and answers it."""
    import re

    def fake(model, idx, max_new_tokens, temperature=0.0, **kw):
        text = tok.decode(idx[0].tolist())
        key = re.search(r"pass key is (\d+)", text).group(1)
        ans = torch.tensor([tok.encode(f" {key}.")], dtype=torch.long)
        return torch.cat([idx, ans], dim=1)

    return fake


def test_run_cell_scores_echoing_model_perfect(tok, monkeypatch):
    monkeypatch.setattr(ep, "generate_cached", _echo_generate(tok))
    cell = ep.run_cell(object(), tok, "cpu", length=128, depth=0.5, n=4, seed=0, max_new=12)
    assert cell["length"] == 128 and cell["depth"] == 0.5 and cell["n"] == 4
    assert cell["correct"] == 4 and cell["acc"] == 1.0
    assert len({s["key"] for s in cell["samples"]}) > 1  # keys vary across samples


def test_run_cell_scores_garbage_model_zero(tok, monkeypatch):
    def fake(model, idx, max_new_tokens, temperature=0.0, **kw):
        ans = torch.tensor([tok.encode(" I forgot everything")], dtype=torch.long)
        return torch.cat([idx, ans], dim=1)

    monkeypatch.setattr(ep, "generate_cached", fake)
    cell = ep.run_cell(object(), tok, "cpu", length=128, depth=0.1, n=3, seed=0, max_new=12)
    assert cell["correct"] == 0 and cell["acc"] == 0.0


def test_format_table_has_grid(tok):
    cells = [
        {"length": 512, "depth": 0.1, "n": 10, "correct": 10, "acc": 1.0, "samples": []},
        {"length": 512, "depth": 0.5, "n": 10, "correct": 9, "acc": 0.9, "samples": []},
        {"length": 2048, "depth": 0.1, "n": 10, "correct": 0, "acc": 0.0, "samples": []},
        {"length": 2048, "depth": 0.5, "n": 10, "correct": 1, "acc": 0.1, "samples": []},
    ]
    table = ep.format_table(cells)
    assert "512" in table and "2048" in table
    assert "1.00" in table and "0.90" in table and "0.10" in table
