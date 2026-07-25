"""scripts/convert_gqa.py: MHA -> GQA checkpoint conversion via the Ainslie et al. (2023)
mean-pool recipe. K/V head projections are mean-pooled into n_kv_head groups; Q and every
other weight are untouched. Loaded via importlib since scripts/ isn't a package."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from microlab.model.reference.variants import VariantConfig, VariantGPT
from microlab.train.config import RunConfig
from microlab.train.trainer import TensorData, Trainer

_SPEC = importlib.util.spec_from_file_location(
    "convert_gqa_script", Path(__file__).resolve().parents[2] / "scripts" / "convert_gqa.py")
convert_gqa = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(convert_gqa)


def _vcfg(n_kv_head=None):
    return VariantConfig(vocab_size=64, block_size=32, n_layer=2, n_head=6, n_embd=48,
                         norm="rms", pos="rope", mlp="swiglu", n_kv_head=n_kv_head)


def test_pool_heads_group_means_exact():
    # 4 heads of dim 2 -> 2 kv heads: rows of each pooled head are the exact arithmetic
    # mean of its group's rows, position by position. Hand-built so the math is auditable.
    n_head, head_dim, C = 4, 2, 3
    w = torch.arange(n_head * head_dim * C, dtype=torch.float64).reshape(n_head * head_dim, C)
    pooled = convert_gqa.pool_heads(w, n_head=n_head, n_kv_head=2)
    assert pooled.shape == (2 * head_dim, C)
    heads = w.reshape(n_head, head_dim, C)
    assert torch.equal(pooled[:head_dim], (heads[0] + heads[1]) / 2)
    assert torch.equal(pooled[head_dim:], (heads[2] + heads[3]) / 2)
    # deterministic: same input -> bit-identical output
    assert torch.equal(pooled, convert_gqa.pool_heads(w, n_head=n_head, n_kv_head=2))


def test_pool_heads_on_bias_vector():
    b = torch.arange(8, dtype=torch.float32)  # 4 heads x head_dim 2
    pooled = convert_gqa.pool_heads(b, n_head=4, n_kv_head=2)
    assert torch.equal(pooled, torch.tensor([1.0, 2.0, 5.0, 6.0]))


def test_convert_state_dict_q_and_rest_untouched():
    torch.manual_seed(0)
    mha = VariantGPT(_vcfg())
    sd = mha.state_dict()
    new = convert_gqa.convert_state_dict(sd, n_head=6, n_embd=48, n_kv_head=2)
    C = 48
    for i in range(2):
        p = f"transformer.h.{i}.attn."
        assert torch.equal(new[p + "q_proj.weight"], sd[p + "c_attn.weight"][:C])
        assert torch.equal(new[p + "q_proj.bias"], sd[p + "c_attn.bias"][:C])
        assert torch.equal(new[p + "c_proj.weight"], sd[p + "c_proj.weight"])
        assert torch.equal(new[p + "c_proj.bias"], sd[p + "c_proj.bias"])
        assert new[p + "kv_proj.weight"].shape == (2 * 2 * 8, C)
        assert new[p + "kv_proj.bias"].shape == (2 * 2 * 8,)
        assert p + "c_attn.weight" not in new
        assert p + "c_attn.bias" not in new
    # every non-attention tensor is carried over bit-identically
    for k in sd:
        if ".attn." not in k:
            assert torch.equal(new[k], sd[k]), k
    # and the result loads cleanly into a GQA model (strict: shapes must line up)
    VariantGPT(_vcfg(n_kv_head=2)).load_state_dict(new)


def test_groups_of_one_is_identity():
    # n_kv_head == n_head: every "group mean" is a single head, so conversion is exact —
    # the GQA model must reproduce the MHA logits (numerics: fused vs split matmul).
    torch.manual_seed(1)
    mha = VariantGPT(_vcfg()).eval()
    new = convert_gqa.convert_state_dict(mha.state_dict(), n_head=6, n_embd=48, n_kv_head=6)
    gqa = VariantGPT(_vcfg(n_kv_head=6)).eval()
    gqa.load_state_dict(new)
    x = torch.randint(0, 64, (2, 16), generator=torch.Generator().manual_seed(2))
    la, _ = mha(x)
    lb, _ = gqa(x)
    assert torch.allclose(la, lb, atol=1e-4)


def test_identical_heads_within_groups_pool_losslessly():
    # THE layout-pinning test: overwrite each group of K/V heads with identical copies
    # (head-major blocks — the layout both attention paths' .view(B,T,heads,head_dim)
    # consume). Mean-pooling identical heads is lossless, and sharing that K/V across the
    # group's query heads is mathematically identical to MHA — so the converted model
    # must reproduce the original logits exactly (up to fused-vs-split matmul numerics).
    # A head-boundary/interleave/packing bug anywhere in the conversion breaks this.
    torch.manual_seed(5)
    n_head, n_kv, C = 6, 2, 48
    head_dim, group = C // n_head, n_head // n_kv
    mha = VariantGPT(_vcfg()).eval()
    with torch.no_grad():
        for block in mha.transformer.h:
            w = block.attn.c_attn.weight  # rows: Q | K | V, each C rows, head-major
            b = block.attn.c_attn.bias
            for base in (C, 2 * C):  # K block, then V block
                heads_w = w[base:base + C].view(n_head, head_dim, C)
                heads_b = b[base:base + C].view(n_head, head_dim)
                for g in range(n_kv):
                    heads_w[g * group:(g + 1) * group] = heads_w[g * group].clone()
                    heads_b[g * group:(g + 1) * group] = heads_b[g * group].clone()
    new = convert_gqa.convert_state_dict(mha.state_dict(), n_head=n_head, n_embd=C,
                                         n_kv_head=n_kv)
    gqa = VariantGPT(_vcfg(n_kv_head=n_kv)).eval()
    gqa.load_state_dict(new)
    x = torch.randint(0, 64, (2, 16), generator=torch.Generator().manual_seed(6))
    la, _ = mha(x)
    lb, _ = gqa(x)
    assert torch.allclose(la, lb, atol=1e-4), f"max diff {(la - lb).abs().max()}"


def test_pooled_model_finite_logits_and_reasonable_kl():
    # Pooling is lossy by design: assert finite logits and a finite, non-negative KL to
    # the original on a fixed batch, and report it — no tight bound.
    torch.manual_seed(3)
    mha = VariantGPT(_vcfg()).eval()
    new = convert_gqa.convert_state_dict(mha.state_dict(), n_head=6, n_embd=48, n_kv_head=2)
    gqa = VariantGPT(_vcfg(n_kv_head=2)).eval()
    gqa.load_state_dict(new)
    x = torch.randint(0, 64, (4, 32), generator=torch.Generator().manual_seed(4))
    logits, _ = gqa(x)
    assert torch.isfinite(logits).all()
    kl = convert_gqa.mean_kl(mha, gqa, x)
    assert math.isfinite(kl) and kl >= 0.0
    print(f"tiny-model conversion KL(orig||pooled) = {kl:.4f} nats/token")


def test_real_dims_shape_only():
    # The actual 1B geometry: 14 heads x head_dim 128, n_kv_head=2 -> 7 heads/group.
    n_head, n_embd, n_kv = 14, 1792, 2
    head_dim = n_embd // n_head
    sd = {
        "transformer.h.0.attn.c_attn.weight": torch.randn(3 * n_embd, n_embd),
        "transformer.h.0.attn.c_attn.bias": torch.randn(3 * n_embd),
        "transformer.h.0.attn.c_proj.weight": torch.randn(n_embd, n_embd),
        "transformer.h.0.attn.c_proj.bias": torch.randn(n_embd),
    }
    new = convert_gqa.convert_state_dict(sd, n_head=n_head, n_embd=n_embd, n_kv_head=n_kv)
    assert new["transformer.h.0.attn.q_proj.weight"].shape == (n_embd, n_embd)
    assert new["transformer.h.0.attn.kv_proj.weight"].shape == (2 * n_kv * head_dim, n_embd)
    assert new["transformer.h.0.attn.kv_proj.bias"].shape == (2 * n_kv * head_dim,)
    # spot-check the 7-head grouping on K: pooled head 0 == mean of source heads 0..6
    kw = sd["transformer.h.0.attn.c_attn.weight"][n_embd:2 * n_embd]
    expect = kw.reshape(n_head, head_dim, n_embd)[:7].mean(dim=0)
    assert torch.equal(new["transformer.h.0.attn.kv_proj.weight"][:head_dim], expect)


def test_scale_correct_multiplies_pooled_by_sqrt_group():
    # 4 heads -> 2 groups of 2: scale-corrected pooling is plain mean-pooling times
    # sqrt(group_size), on weights and biases alike.
    w = torch.randn(8, 3, generator=torch.Generator().manual_seed(10))
    plain = convert_gqa.pool_heads(w, n_head=4, n_kv_head=2)
    sc = convert_gqa.pool_heads(w, n_head=4, n_kv_head=2, scale_correct=True)
    assert torch.allclose(sc, plain * math.sqrt(2))


def test_scale_correct_restores_rms_for_orthogonal_heads():
    # Two mutually orthogonal heads (head_dim 1): the mean shrinks the row norm by
    # 1/sqrt(2); the correction restores it to the original per-head norm exactly.
    w = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    sc = convert_gqa.pool_heads(w, n_head=2, n_kv_head=1, scale_correct=True)
    assert float(sc.norm()) == pytest.approx(1.0)


def test_convert_state_dict_scale_correct_only_touches_kv():
    torch.manual_seed(12)
    mha = VariantGPT(_vcfg())
    sd = mha.state_dict()
    plain = convert_gqa.convert_state_dict(sd, n_head=6, n_embd=48, n_kv_head=2)
    sc = convert_gqa.convert_state_dict(sd, n_head=6, n_embd=48, n_kv_head=2,
                                        scale_correct=True)
    factor = math.sqrt(3)  # 6 heads / 2 kv heads = groups of 3
    for i in range(2):
        p = f"transformer.h.{i}.attn."
        assert torch.equal(sc[p + "q_proj.weight"], plain[p + "q_proj.weight"])
        assert torch.equal(sc[p + "c_proj.weight"], plain[p + "c_proj.weight"])
        assert torch.allclose(sc[p + "kv_proj.weight"], plain[p + "kv_proj.weight"] * factor)
        assert torch.allclose(sc[p + "kv_proj.bias"], plain[p + "kv_proj.bias"] * factor)
    for k in sd:
        if ".attn." not in k:
            assert torch.equal(sc[k], sd[k]), k


def test_convert_checkpoint_records_scale_correct(tmp_path):
    data = TensorData(torch.randint(0, 64, (2000,), generator=torch.Generator().manual_seed(9)))
    tr = Trainer(_tiny_run_cfg(out_dir=str(tmp_path / "run")), data)
    src = tmp_path / "run" / "ckpt_0.pt"
    tr.save_checkpoint(str(src))
    out = tmp_path / "sc.pt"
    convert_gqa.convert_checkpoint(str(src), str(out), n_kv_head=1, scale_correct=True)
    ckpt = torch.load(str(out), map_location="cpu", weights_only=False)
    assert ckpt["conversion"]["scale_correct"] is True
    assert "scale" in ckpt["conversion"]["method"]


def test_n_kv_head_must_divide_n_head():
    sd = {"transformer.h.0.attn.c_attn.weight": torch.randn(9, 3)}
    with pytest.raises(ValueError, match="divide"):
        convert_gqa.convert_state_dict(sd, n_head=3, n_embd=3, n_kv_head=2)


def _tiny_run_cfg(**kw):
    base = dict(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32,
                warmup_steps=1, max_steps=2, lr_decay_steps=2, batch_size=4,
                eval_interval=1000, ckpt_interval=1000, device="cpu", dtype="float32")
    base.update(kw)
    return RunConfig(**base)


def test_convert_checkpoint_end_to_end(tmp_path):
    # Full pipeline on a real trainer checkpoint: output has model weights + updated cfg,
    # NO optimizer state, and warm-starts a GQA Trainer built from the run config.
    data = TensorData(torch.randint(0, 64, (2000,), generator=torch.Generator().manual_seed(9)))
    tr = Trainer(_tiny_run_cfg(out_dir=str(tmp_path / "run")), data)
    tr.train()
    src = tmp_path / "run" / "ckpt_2.pt"
    tr.save_checkpoint(str(src))
    out = tmp_path / "converted.pt"
    summary = convert_gqa.convert_checkpoint(str(src), str(out), n_kv_head=1)
    assert summary["params_after"] < summary["params_before"]
    ckpt = torch.load(str(out), map_location="cpu", weights_only=False)
    assert "optimizer" not in ckpt
    assert ckpt["cfg"].n_kv_head == 1
    assert ckpt["cfg"].n_head == 2  # everything else preserved
    assert ckpt["converted_from"]["step"] == 2
    # run-cfg-wins: build the uptrain trainer from its own config; strict load asserts
    # shape compatibility with the pooled weights.
    up_cfg = _tiny_run_cfg(n_kv_head=1, out_dir=str(tmp_path / "up"))
    up = Trainer(up_cfg, data)
    up.raw_model.load_state_dict(ckpt["model"])


def test_convert_checkpoint_rejects_non_mha(tmp_path):
    # Converting an already-GQA checkpoint would silently re-pool; raise instead.
    data = TensorData(torch.randint(0, 64, (2000,), generator=torch.Generator().manual_seed(9)))
    tr = Trainer(_tiny_run_cfg(n_kv_head=1, out_dir=str(tmp_path / "run")), data)
    src = tmp_path / "run" / "ckpt_0.pt"
    tr.save_checkpoint(str(src))
    with pytest.raises(ValueError, match="MHA"):
        convert_gqa.convert_checkpoint(str(src), str(tmp_path / "x.pt"), n_kv_head=1)
