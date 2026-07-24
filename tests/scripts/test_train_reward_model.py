"""scripts/train_reward_model.py: the deterministic holdout split (seeded shuffle, LAST k held
out, no train overlap), sequence building (prompt+response+sentinel, prompt-side truncation,
counted skips), and a tiny end-to-end CPU run that learns a separable preference set, writes
holdout.jsonl + eval.json, and keeps exactly one (the best) checkpoint. Loaded via importlib
since scripts/ isn't a package."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from microlab.model.reference.sft import format_chat
from microlab.model.reference.variants import VariantConfig, VariantGPT
from microlab.tokenizer.fast import FastTokenizer
from microlab.train.reward import load_reward_checkpoint

_SPEC = importlib.util.spec_from_file_location(
    "train_reward_model_script",
    Path(__file__).resolve().parents[2] / "scripts" / "train_reward_model.py")
trm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(trm)


class _ByteTok:
    """Byte-level tokenizer: encode is exact per char so truncation boundaries don't shift."""

    def encode(self, s):
        return list(s.encode("utf-8"))


# ---------------------------------------------------------------- split_holdout


def test_split_holdout_is_deterministic_and_disjoint():
    train_a, hold_a = trm.split_holdout(100, holdout=17, seed=1337)
    train_b, hold_b = trm.split_holdout(100, holdout=17, seed=1337)
    assert train_a == train_b and hold_a == hold_b  # same seed -> identical split
    assert len(hold_a) == 17 and len(train_a) == 83
    assert set(train_a).isdisjoint(hold_a)          # never trained on
    assert set(train_a) | set(hold_a) == set(range(100))
    # The holdout is the LAST k of the seeded shuffle (so train+holdout is one permutation).
    perm = train_a + hold_a
    assert sorted(perm) == list(range(100)) and perm != list(range(100))


def test_split_holdout_seed_changes_split():
    _, hold_a = trm.split_holdout(100, holdout=17, seed=1337)
    _, hold_b = trm.split_holdout(100, holdout=17, seed=7)
    assert hold_a != hold_b


def test_split_holdout_rejects_bad_sizes():
    with pytest.raises(ValueError, match="holdout"):
        trm.split_holdout(10, holdout=10, seed=0)
    with pytest.raises(ValueError, match="holdout"):
        trm.split_holdout(10, holdout=0, seed=0)


# ---------------------------------------------------------------- build_reward_sequences


def test_build_reward_sequences_appends_sentinel_and_keeps_pairs():
    tok = _ByteTok()
    prompt, _ = format_chat("Say hi")
    rows = [{"prompt": prompt, "chosen": "good", "rejected": "bad"}]
    pairs, skipped = trm.build_reward_sequences(tok, rows, block_size=256)
    assert skipped == 0 and len(pairs) == 1
    chosen_ids, rejected_ids = pairs[0]
    assert chosen_ids == tok.encode(prompt + "good" + trm.END_SENTINEL)
    assert rejected_ids == tok.encode(prompt + "bad" + trm.END_SENTINEL)


def test_build_reward_sequences_truncates_prompt_from_left():
    tok = _ByteTok()
    rows = [{"prompt": "P" * 100, "chosen": "c" * 10, "rejected": "r" * 5}]
    block = 32
    pairs, skipped = trm.build_reward_sequences(tok, rows, block_size=block)
    assert skipped == 0
    chosen_ids, rejected_ids = pairs[0]
    # Response (+ sentinel) is kept whole; the prompt loses its LEFT side to fit the block.
    resp_c = tok.encode("c" * 10 + trm.END_SENTINEL)
    resp_r = tok.encode("r" * 5 + trm.END_SENTINEL)
    assert len(chosen_ids) == block and len(rejected_ids) == block
    assert chosen_ids[-len(resp_c):] == resp_c
    assert rejected_ids[-len(resp_r):] == resp_r
    assert chosen_ids[:-len(resp_c)] == tok.encode("P" * 100)[-(block - len(resp_c)):]


def test_build_reward_sequences_skips_and_counts_oversized_responses():
    tok = _ByteTok()
    rows = [
        {"prompt": "p", "chosen": "x" * 100, "rejected": "ok"},  # chosen can't fit -> skip pair
        {"prompt": "p", "chosen": "ok", "rejected": "fine"},
    ]
    pairs, skipped = trm.build_reward_sequences(tok, rows, block_size=32)
    assert skipped == 1 and len(pairs) == 1


# ---------------------------------------------------------------- end-to-end tiny run


def _tiny_sft_run(tmp_path):
    """Write a tiny servable SFT-style run (ckpt + tokenizer) and return (run_dir, tok_path)."""
    tok = FastTokenizer.train(
        ["a correct and helpful answer", "wrong", "question words here"] * 4,
        vocab_size=300, save_path=str(tmp_path / "tokenizer.json"))
    torch.manual_seed(0)
    cfg = VariantConfig(vocab_size=tok.vocab_size, block_size=128, n_layer=2, n_head=2,
                        n_embd=16, norm="rms", pos="rope", mlp="swiglu")
    run_dir = tmp_path / "sft-run"
    run_dir.mkdir()
    torch.save({"model": VariantGPT(cfg).state_dict(), "step": 100, "cfg": cfg},
               run_dir / "ckpt_100.pt")
    (run_dir / "tokenizer.json").write_text((tmp_path / "tokenizer.json").read_text())
    return run_dir, run_dir / "tokenizer.json"


def _write_prefs(path, n):
    prefs = [{"prompt": format_chat(f"question {i}")[0],
              "chosen": "a correct and helpful answer",
              "rejected": "wrong"} for i in range(n)]
    path.write_text("\n".join(json.dumps(x) for x in prefs) + "\n")
    return prefs


def test_run_train_reward_end_to_end(tmp_path):
    run_dir, tok_path = _tiny_sft_run(tmp_path)
    prefs_path = tmp_path / "prefs.jsonl"
    prefs = _write_prefs(prefs_path, 12)
    uf_path = tmp_path / "uf.jsonl"
    _write_prefs(uf_path, 5)

    out = tmp_path / "rm-run"
    result = trm.run_train_reward(
        base_ckpt=run_dir, prefs=prefs_path, out=out, tokenizer=tok_path, uf_prefs=uf_path,
        epochs=6, lr=1e-3, batch_size=2, grad_accum=2, holdout=3, uf_n=4, seed=1337,
        device="cpu", log_interval=1)

    # Separable synthetic prefs: the tiny RM must learn to rank chosen above rejected.
    assert result["train_acc_final"] > 0.5
    assert math.isfinite(result["eval"]["holdout_acc"])

    # holdout.jsonl: exactly the 3 held-out pairs, tagged with their original indices, and
    # reproducible from the split (so future evals reuse the exact split).
    hold_rows = [json.loads(li) for li in (out / "holdout.jsonl").read_text().splitlines()]
    _, hold_idx = trm.split_holdout(12, holdout=3, seed=1337)
    assert [r["index"] for r in hold_rows] == hold_idx
    assert all(r["chosen"] == prefs[r["index"]]["chosen"] for r in hold_rows)

    # eval.json: the required report keys, with sane values.
    ev = json.loads((out / "eval.json").read_text())
    for key in ("holdout_acc", "holdout_margin_mean", "uf_acc", "n_holdout", "n_uf",
                "trained_pairs", "epochs", "lr", "kept_epoch"):
        assert key in ev, key
    assert ev["n_holdout"] == 3 and ev["n_uf"] == 4
    assert ev["trained_pairs"] == 9 and ev["epochs"] == 6 and ev["lr"] == 1e-3
    assert 1 <= ev["kept_epoch"] <= 6
    assert 0.0 <= ev["holdout_acc"] <= 1.0 and 0.0 <= ev["uf_acc"] <= 1.0

    # Exactly ONE checkpoint kept (the best epoch's), loadable as a RewardModel.
    ckpts = sorted(out.glob("ckpt_*.pt"))
    assert len(ckpts) == 1
    model, step = load_reward_checkpoint(ckpts[0], device="cpu")
    assert step == ev["kept_step"]
    assert model.backbone.config.vocab_size == FastTokenizer.load(str(tok_path)).vocab_size

    # Progressive logging: one train_log.jsonl line per optimizer step plus epoch evals.
    log_lines = [json.loads(li) for li in (out / "train_log.jsonl").read_text().splitlines()]
    step_lines = [li for li in log_lines if li["kind"] == "step"]
    eval_lines = [li for li in log_lines if li["kind"] == "epoch_eval"]
    assert len(step_lines) == result["steps"]
    assert len(eval_lines) == 6
    assert (out / "tokenizer.json").exists()


def test_run_train_reward_rejects_holdout_larger_than_data(tmp_path):
    run_dir, tok_path = _tiny_sft_run(tmp_path)
    prefs_path = tmp_path / "prefs.jsonl"
    _write_prefs(prefs_path, 4)
    with pytest.raises(ValueError, match="holdout"):
        trm.run_train_reward(
            base_ckpt=run_dir, prefs=prefs_path, out=tmp_path / "o", tokenizer=tok_path,
            uf_prefs=prefs_path, epochs=1, holdout=4, device="cpu")
