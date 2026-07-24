"""scripts/eval_best_of_n.py pure logic: the RM scoring-sequence construction (pinned to the
EXACT training-time construction in scripts/train_reward_model.py — a silent mismatch there
would make every score wrong), best-of-n selection, the usable-row filter, padding-invariant
batched scoring, and outcome aggregation. Generation and codex calls aren't exercised.
Loaded via importlib since scripts/ isn't a package."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from microlab.model.reference.sft import format_chat
from microlab.model.reference.variants import VariantConfig, VariantGPT
from microlab.tokenizer.fast import FastTokenizer
from microlab.train.reward import RewardModel

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ebn = _load("eval_best_of_n")
trm = _load("train_reward_model")


class _ByteTok:
    """Byte-level tokenizer: encode is exact per char so truncation boundaries don't shift."""

    def encode(self, s):
        return list(s.encode("utf-8"))


# ---------------------------------------------------------------- build_candidate_sequences


def test_scoring_sequences_match_training_construction_exactly():
    """The invariant this whole eval rests on: candidate sequences are byte-for-byte what
    build_reward_sequences produced at TRAINING time — prompt_ids + encode(resp + sentinel),
    encoded SEPARATELY then concatenated (one-string encoding would merge BPE boundaries)."""
    tok = FastTokenizer.train(["what is the capital of France? Paris, obviously."] * 4,
                              vocab_size=300)
    prompt, _ = format_chat("what is the capital?")
    cands = ["Paris", "the capital is Paris, obviously", ""]
    seqs = ebn.build_candidate_sequences(tok, prompt, cands, block_size=1024)
    assert len(seqs) == 3
    for cand, seq in zip(cands, seqs, strict=True):
        # Pin against the training-time constructor itself...
        train_pairs, _ = trm.build_reward_sequences(
            tok, [{"prompt": prompt, "chosen": cand, "rejected": cand}], 1024)
        assert seq == train_pairs[0][0]
        # ...and against the explicit construction, so a drift in EITHER is caught loudly.
        assert seq == tok.encode(prompt) + tok.encode(cand + trm.END_SENTINEL)


def test_scoring_sequences_truncate_prompt_from_left_like_training():
    tok = _ByteTok()
    prompt = "P" * 100
    seqs = ebn.build_candidate_sequences(tok, prompt, ["c" * 10], block_size=32)
    resp = tok.encode("c" * 10 + trm.END_SENTINEL)
    assert len(seqs[0]) == 32
    assert seqs[0][-len(resp):] == resp                     # response + sentinel never cut
    assert seqs[0][:-len(resp)] == tok.encode(prompt)[-(32 - len(resp)):]  # left-truncated


def test_scoring_sequences_raise_when_response_fills_block():
    # Training SKIPS such pairs; here a skip would silently misalign candidate indices with
    # scores, so it must raise instead.
    with pytest.raises(ValueError, match="fill"):
        ebn.build_candidate_sequences(_ByteTok(), "p", ["ok", "x" * 100], block_size=32)


# ---------------------------------------------------------------- pick_best


def test_pick_best_is_argmax_first_on_ties():
    assert ebn.pick_best([0.1, 3.0, -2.0, 3.0]) == 1  # max, earliest index wins ties
    assert ebn.pick_best([-5.0]) == 0
    assert ebn.pick_best([2.0, 2.0]) == 0
    with pytest.raises(ValueError, match="empty"):
        ebn.pick_best([])


# ---------------------------------------------------------------- score_sequences


def _tiny_rm(vocab: int = 300, block: int = 64) -> RewardModel:
    torch.manual_seed(0)
    cfg = VariantConfig(vocab_size=vocab, block_size=block, n_layer=2, n_head=2, n_embd=16,
                        dropout=0.0, norm="rms", pos="rope", mlp="swiglu")
    return RewardModel(VariantGPT(cfg)).eval()


def test_score_sequences_batched_equals_individual():
    """Right-padding a mixed-length batch must not change any score: batched scoring equals
    scoring each sequence alone (catches lengths/padding bugs — the classic way a reward
    eval goes silently wrong)."""
    rm = _tiny_rm()
    seqs = [[5, 6, 7], [8, 9, 10, 11, 12, 13], [14], [1, 2, 3, 4, 5, 6, 7, 8, 9]]
    batched = ebn.score_sequences(rm, seqs, device="cpu", use_amp=False)
    solo = [ebn.score_sequences(rm, [s], device="cpu", use_amp=False)[0] for s in seqs]
    assert len(batched) == 4
    for b, s in zip(batched, solo, strict=True):
        assert abs(b - s) < 1e-4


def test_score_sequences_depends_on_last_token():
    rm = _tiny_rm()
    a, b = [5, 6, 7, 8], [5, 6, 7, 9]
    sa, sb = ebn.score_sequences(rm, [a, b], device="cpu", use_amp=False)
    assert sa != sb  # scored at the last real token, so it must react to it


# ---------------------------------------------------------------- select_usable_rows


def test_select_usable_rows_skips_prefix_and_applies_block_guard():
    tok = _ByteTok()
    rows = [{"instruction": f"i{n}", "context": ""} for n in range(10)]
    rows[6]["context"] = "X" * 500  # templated prompt too long for prompt+max_new<=block
    picked, skipped_long = ebn.select_usable_rows(tok, rows, skip=5, limit=3,
                                                  max_new=16, block_size=256)
    assert [p[0] for p in picked] == [5, 7, 8]  # absolute row indices; row 6 skipped
    assert skipped_long == 1
    for idx, instruction, prompt in picked:
        assert instruction == f"i{idx}"
        assert prompt == format_chat(instruction)[0]
        assert len(tok.encode(prompt)) + 16 <= 256


def test_select_usable_rows_raises_when_data_runs_out():
    rows = [{"instruction": "hi", "context": ""} for _ in range(6)]
    with pytest.raises(ValueError, match="usable"):
        ebn.select_usable_rows(_ByteTok(), rows, skip=5, limit=3, max_new=16, block_size=256)


# ---------------------------------------------------------------- aggregate


def _item(outcome, scores, best_idx):
    return {"row": 0, "instruction": "q", "a": "A", "b": "B",
            "scores": scores, "best_idx": best_idx, "outcome": outcome}


def test_aggregate_counts_and_win_rate():
    items = [_item("A", [1.0, 0.0, 3.0], 2),
             _item("A", [0.5, 4.0, 1.0], 1),
             _item("B", [0.0, 2.0, 1.0], 1),
             _item("tie", [1.0, 1.0, 1.0], 0)]
    rep = ebn.aggregate(items)
    assert rep["a_wins"] == 2 and rep["b_wins"] == 1 and rep["ties"] == 1
    assert rep["decided"] == 3
    assert rep["win_rate_best_of_n"] == pytest.approx(2 / 3)


def test_aggregate_rm_score_stats():
    items = [_item("A", [1.0, 0.0, 3.0], 2),   # picked 3.0, first 1.0, spread 3.0
             _item("B", [0.0, 4.0, 2.0], 1)]   # picked 4.0, first 0.0, spread 4.0
    rep = ebn.aggregate(items)
    assert rep["rm_scores"]["mean_picked"] == pytest.approx(3.5)
    assert rep["rm_scores"]["mean_first_sample"] == pytest.approx(0.5)
    assert rep["rm_scores"]["mean_spread"] == pytest.approx(3.5)


def test_aggregate_rm_judge_agreement_excludes_no_preference_items():
    # A is the argmax candidate, so on decided items the judge "agrees" with the RM exactly
    # when A wins — EXCEPT items where best_idx == 0 (A is literally the first sample; the RM
    # expressed no preference between A and B, so they can't count either way).
    items = [_item("A", [1.0, 0.0, 3.0], 2),    # agree
             _item("B", [0.0, 2.0, 1.0], 1),    # disagree
             _item("A", [5.0, 1.0, 0.0], 0),    # best is the first sample -> excluded
             _item("tie", [1.0, 0.0, 3.0], 2)]  # undecided -> excluded
    rep = ebn.aggregate(items)
    agr = rep["rm_judge_agreement"]
    assert agr["n"] == 2 and agr["agree"] == 1
    assert agr["rate"] == pytest.approx(0.5)


def test_aggregate_win_rate_none_when_all_tied():
    rep = ebn.aggregate([_item("tie", [1.0, 2.0], 1)])
    assert rep["decided"] == 0 and rep["win_rate_best_of_n"] is None
    assert rep["rm_judge_agreement"]["n"] == 0
    assert rep["rm_judge_agreement"]["rate"] is None


# ---------------------------------------------------------------- ensure_codex


def test_ensure_codex_raises_when_missing(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    with pytest.raises(RuntimeError, match="codex"):
        ebn.ensure_codex()
