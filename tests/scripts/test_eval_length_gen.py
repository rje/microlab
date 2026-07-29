"""scripts/eval_length_gen.py pure logic: batch planning, per-position NLL accumulation
and summarization (full-window mean + bucketed means), the stubbed loss loop, and the
checkpoint-rebuild path for BOTH pos variants (rope: cache extended at native theta;
nope: nothing to extend, just a bigger block_size). Loaded via importlib since scripts/
isn't a package; no GPU."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from microlab.model.reference.variants import VariantConfig, VariantGPT
from microlab.train.config import RunConfig

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


elg = _load("eval_length_gen")


# ------------------------------------------------------------------------ batch planning

def test_seqs_for_budget_is_ceil_division():
    assert elg.seqs_for_budget(1000, 128) == 8      # 7.8 -> 8
    assert elg.seqs_for_budget(1024, 128) == 8      # exact
    assert elg.seqs_for_budget(4096, 4096) == 1
    assert elg.seqs_for_budget(1, 4096) == 1        # never zero sequences


def test_micro_batch_size_caps_tokens_per_forward():
    assert elg.micro_batch_size(512, 65536) == 128
    assert elg.micro_batch_size(4096, 65536) == 16
    assert elg.micro_batch_size(100000, 65536) == 1  # longer than the cap -> one at a time


def test_batch_plan_covers_exactly_n_seqs():
    assert elg.batch_plan(10, 4) == [4, 4, 2]
    assert elg.batch_plan(4, 4) == [4]
    assert elg.batch_plan(3, 8) == [3]
    with pytest.raises(ValueError):
        elg.batch_plan(0, 4)


# --------------------------------------------------------------- per-position NLL logic

def test_position_nll_sums_match_manual_cross_entropy():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(3, 5, 11, generator=g)
    targets = torch.randint(0, 11, (3, 5), generator=g)
    sums = elg.position_nll_sums(logits, targets)
    assert sums.shape == (5,) and sums.dtype == torch.float64
    manual = F.cross_entropy(
        logits.reshape(-1, 11), targets.reshape(-1), reduction="none"
    ).reshape(3, 5).sum(0)
    assert torch.allclose(sums, manual.double(), atol=1e-6)


def test_position_nll_sums_accumulate_across_batches():
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(4, 6, 7, generator=g)
    targets = torch.randint(0, 7, (4, 6), generator=g)
    whole = elg.position_nll_sums(logits, targets)
    split = (elg.position_nll_sums(logits[:2], targets[:2])
             + elg.position_nll_sums(logits[2:], targets[2:]))
    assert torch.allclose(whole, split, atol=1e-6)


def test_summarize_positions_mean_ppl_and_buckets():
    # 2 sequences, T=8, bucket=4; per-position MEAN loss = position index / 10
    means = torch.arange(8, dtype=torch.float64) / 10
    sums = means * 2
    out = elg.summarize_positions(sums, n_seqs=2, bucket=4)
    expected_mean = means.mean().item()
    assert out["mean_loss"] == pytest.approx(expected_mean)
    assert out["ppl"] == pytest.approx(math.exp(expected_mean))
    assert out["bucket_size"] == 4
    assert out["bucket_means"] == pytest.approx([means[:4].mean().item(),
                                                 means[4:].mean().item()])


def test_summarize_positions_rejects_partial_buckets():
    with pytest.raises(ValueError, match="bucket"):
        elg.summarize_positions(torch.zeros(10, dtype=torch.float64), n_seqs=1, bucket=4)


# ------------------------------------------------------------------- stubbed loss loop

class _UniformModel:
    """Stub model: uniform logits -> per-token NLL is exactly ln(vocab)."""

    def __init__(self, vocab: int) -> None:
        self.vocab = vocab

    def __call__(self, x):
        return torch.zeros(x.shape[0], x.shape[1], self.vocab), None


class _RecordingData:
    """ShardDataset-shaped stub that records get_batch calls."""

    def __init__(self, vocab: int) -> None:
        self.vocab = vocab
        self.calls: list[tuple[int, int]] = []

    def get_batch(self, block_size, batch_size, device="cpu", generator=None):
        assert generator is not None, "eval must draw from a seeded generator"
        self.calls.append((block_size, batch_size))
        x = torch.randint(0, self.vocab, (batch_size, block_size), generator=generator)
        y = torch.randint(0, self.vocab, (batch_size, block_size), generator=generator)
        return x, y


def test_eval_loss_at_length_uniform_model_gives_log_vocab():
    data = _RecordingData(vocab=13)
    out = elg.eval_loss_at_length(_UniformModel(13), data, length=8, n_seqs=10,
                                  micro_bs=4, device="cpu", seed=0, bucket=4)
    assert out["length"] == 8
    assert out["n_seqs"] == 10 and out["tokens"] == 80
    assert out["mean_loss"] == pytest.approx(math.log(13), abs=1e-6)
    assert out["ppl"] == pytest.approx(13.0, abs=1e-4)
    assert out["bucket_means"] == pytest.approx([math.log(13)] * 2, abs=1e-6)
    # windows are drawn AT the eval length, in micro-batches covering exactly n_seqs
    assert data.calls == [(8, 4), (8, 4), (8, 2)]


def test_eval_loss_at_length_is_deterministic_per_seed():
    torch.manual_seed(123)
    model = VariantGPT(VariantConfig(vocab_size=32, block_size=16, n_layer=1, n_head=2,
                                     n_embd=16, norm="rms", pos="nope", mlp="swiglu"))
    model.eval()
    a = elg.eval_loss_at_length(model, _RecordingData(32), length=8, n_seqs=6,
                                micro_bs=3, device="cpu", seed=7, bucket=8)
    b = elg.eval_loss_at_length(model, _RecordingData(32), length=8, n_seqs=6,
                                micro_bs=3, device="cpu", seed=7, bucket=8)
    assert a["mean_loss"] == b["mean_loss"]


# --------------------------------------------------- checkpoint rebuild for both arms

def _write_ckpt(tmp_path, pos: str) -> Path:
    torch.manual_seed(0)
    cfg = RunConfig(vocab_size=64, block_size=16, n_layer=1, n_head=2, n_embd=16,
                    norm="rms", pos=pos, mlp="swiglu", out_dir=str(tmp_path))
    model = VariantGPT(VariantConfig(
        vocab_size=cfg.vocab_size, block_size=cfg.block_size, n_layer=cfg.n_layer,
        n_head=cfg.n_head, n_embd=cfg.n_embd, norm=cfg.norm, pos=cfg.pos, mlp=cfg.mlp))
    torch.save({"model": model.state_dict(), "cfg": cfg, "step": 10},
               tmp_path / "ckpt_000010.pt")
    return tmp_path


@pytest.mark.parametrize("pos", ["nope", "rope"])
def test_load_for_eval_extends_both_arms_beyond_train_length(tmp_path, pos):
    ep = _load("eval_passkey")
    run_dir = _write_ckpt(tmp_path, pos)
    model, step, cfg, eval_block = ep.load_for_eval(run_dir, min_context=64, device="cpu")
    assert step == 10 and eval_block == 64
    assert model.config.block_size == 64
    x = torch.randint(0, 64, (1, 48))  # 3x the trained window
    logits, _ = model(x)
    assert logits.shape == (1, 48, 64)
    if pos == "rope":
        assert model.transformer.h[0].attn.rope_cos.shape[0] == 64  # cache extended
    else:
        assert all("rope" not in n for n, _ in model.named_buffers())  # nothing to extend
