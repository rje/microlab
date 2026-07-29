"""Gated DeltaNet correctness.

The whole GDN/KDA hybrid ablation rests on `gdn_chunkwise` computing exactly what
`gdn_recurrent` computes. The recurrent form is the definition — a transcription of the
paper's recurrence with no algebra applied — so it is correct by inspection. The
chunkwise form is a nontrivial derivation (cumulative-decay substitution + a
unit-lower-triangular solve) and is exactly the kind of thing that can be subtly wrong
and still train to a plausible-looking loss. These tests are the gate: if they fail, the
ablation verdict is meaningless and must not be run.
"""

import pytest
import torch

from microlab.model.reference.variants import (
    GatedDeltaNet,
    VariantConfig,
    VariantGPT,
    gdn_chunkwise,
    gdn_recurrent,
)


def _rand(B=2, H=3, T=64, Dk=16, Dv=16, seed=0, alpha_lo=0.90, alpha_hi=1.0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, T, Dk, generator=g, dtype=torch.float64)
    k = torch.randn(B, H, T, Dk, generator=g, dtype=torch.float64)
    v = torch.randn(B, H, T, Dv, generator=g, dtype=torch.float64)
    # Keys are L2-normalised in the real module; the delta rule assumes it.
    q = torch.nn.functional.normalize(q, dim=-1)
    k = torch.nn.functional.normalize(k, dim=-1)
    span = alpha_hi - alpha_lo
    alpha = alpha_lo + span * torch.rand(B, H, T, generator=g, dtype=torch.float64)
    beta = torch.rand(B, H, T, generator=g, dtype=torch.float64)
    return q, k, v, alpha, beta


@pytest.mark.parametrize("chunk", [8, 16, 32, 64])
def test_chunkwise_matches_recurrent(chunk):
    """The load-bearing test: same numbers, every chunk length."""
    q, k, v, alpha, beta = _rand(T=64)
    ref = gdn_recurrent(q, k, v, alpha, beta)
    got = gdn_chunkwise(q, k, v, alpha, beta, chunk=chunk)
    assert torch.allclose(ref, got, atol=1e-8, rtol=1e-6), (
        f"chunk={chunk} max abs diff {(ref - got).abs().max().item():.3e}"
    )


def test_matches_across_multiple_chunks():
    """State must carry correctly across chunk boundaries, not just within one chunk."""
    q, k, v, alpha, beta = _rand(T=256, seed=7)
    ref = gdn_recurrent(q, k, v, alpha, beta)
    got = gdn_chunkwise(q, k, v, alpha, beta, chunk=32)
    assert torch.allclose(ref, got, atol=1e-8, rtol=1e-6)


def test_matches_with_strong_decay():
    """Small alpha is the numerically hostile case: beta/A_t grows as A_t decays. If
    this fails, gdn_chunkwise needs a shorter chunk or a log-space rescale."""
    q, k, v, alpha, beta = _rand(T=128, seed=3, alpha_lo=0.5, alpha_hi=0.8)
    ref = gdn_recurrent(q, k, v, alpha, beta)
    got = gdn_chunkwise(q, k, v, alpha, beta, chunk=16)
    assert torch.allclose(ref, got, atol=1e-6, rtol=1e-4)


def test_no_decay_no_delta_is_plain_linear_attention():
    """alpha=1, beta=1 with unit keys reduces the delta rule to a known closed form;
    a sanity anchor independent of both implementations."""
    B, H, T, D = 1, 1, 8, 4
    q, k, v, _, _ = _rand(B=B, H=H, T=T, Dk=D, Dv=D, seed=11)
    alpha = torch.ones(B, H, T, dtype=torch.float64)
    beta = torch.ones(B, H, T, dtype=torch.float64)
    ref = gdn_recurrent(q, k, v, alpha, beta)
    got = gdn_chunkwise(q, k, v, alpha, beta, chunk=4)
    assert torch.allclose(ref, got, atol=1e-8, rtol=1e-6)


def test_causality():
    """Output at t must not depend on inputs after t — the failure a chunkwise mask
    off-by-one produces, and one a loss curve would never reveal."""
    q, k, v, alpha, beta = _rand(T=64, seed=5)
    base = gdn_chunkwise(q, k, v, alpha, beta, chunk=16)
    v2 = v.clone()
    v2[:, :, 40:] += 10.0          # perturb strictly after t=39
    pert = gdn_chunkwise(q, k, v2, alpha, beta, chunk=16)
    assert torch.allclose(base[:, :, :40], pert[:, :, :40], atol=1e-9)
    assert not torch.allclose(base[:, :, 40:], pert[:, :, 40:])


@pytest.mark.parametrize("T", [1, 5, 63, 64, 65, 100])
def test_ragged_sequence_matches_recurrent(T):
    """Generation feeds T=1,2,3,... token by token, so a sequence shorter than (or not a
    multiple of) the chunk MUST work and must be exact, not approximate. Regression test:
    the first GDN smoke run trained fine for 5 steps and then died in end-of-run sample
    generation on T=1. Right-padding is a no-op on the state only if k=0/beta=0/alpha=1
    is handled correctly — this is what proves it."""
    q, k, v, alpha, beta = _rand(T=T, seed=T)
    ref = gdn_recurrent(q, k, v, alpha, beta)
    got = gdn_chunkwise(q, k, v, alpha, beta, chunk=64)
    assert got.shape == ref.shape
    assert torch.allclose(ref, got, atol=1e-8, rtol=1e-6), (
        f"T={T} max abs diff {(ref - got).abs().max().item():.3e}"
    )


def test_incremental_generation_shapes():
    """The exact call pattern sample.py uses: grow the context one token at a time."""
    torch.manual_seed(0)
    m = VariantGPT(_cfg(n_layer=4, hybrid_every=4, gdn_chunk=64, block_size=128))
    idx = torch.randint(0, 64, (1, 1))
    for _ in range(8):
        logits, _ = m(idx)
        assert logits.shape == (1, idx.shape[1], 64)
        idx = torch.cat([idx, torch.randint(0, 64, (1, 1))], dim=1)


@pytest.mark.parametrize("alpha_lo,alpha_hi", [(0.4, 0.6), (0.1, 0.9), (0.01, 0.05)])
def test_fp32_stable_under_hostile_decay(alpha_lo, alpha_hi):
    """REGRESSION for the 2026-07-29 NaN. The original chunkwise form divided by the
    cumulative decay A_t; with alpha~0.5 and chunk=64, A_64 ~ 1e-23 so beta/A_t ~ 1e22.
    The forward pass hid it (the factors cancel) but the backward NaN'd in training.

    The earlier fp64 tests could not catch this — 1e22 is harmless in float64. This runs
    at fp32, the precision training actually uses, at full chunk=64, and checks the
    OUTPUT MAGNITUDE stays on the scale of the inputs rather than only checking for NaN,
    because the cancellation means a broken implementation can still return finite
    numbers on the forward pass."""
    q, k, v, alpha, beta = _rand(T=128, seed=42, alpha_lo=alpha_lo, alpha_hi=alpha_hi)
    q32, k32, v32 = q.float(), k.float(), v.float()
    a32, b32 = alpha.float(), beta.float()
    got = gdn_chunkwise(q32, k32, v32, a32, b32, chunk=64)
    assert torch.isfinite(got).all(), "non-finite output"
    # v is unit-normal, so outputs of order 1e2 already mean the scan is amplifying.
    assert got.abs().max() < 1e2, f"output blew up: absmax {got.abs().max().item():.3e}"
    ref = gdn_recurrent(q, k, v, alpha, beta).float()
    assert torch.allclose(ref, got, atol=1e-4, rtol=1e-3), (
        f"fp32 diverged from fp64 reference: max diff {(ref - got).abs().max().item():.3e}"
    )


def test_gradients_finite_under_hostile_decay():
    """The forward hid the original bug; only the backward exposed it. So check grads."""
    q, k, v, alpha, beta = _rand(T=128, seed=13, alpha_lo=0.3, alpha_hi=0.7)
    q, k, v = (t.float().requires_grad_(True) for t in (q, k, v))
    a, b = alpha.float().requires_grad_(True), beta.float().requires_grad_(True)
    gdn_chunkwise(q, k, v, a, b, chunk=64).sum().backward()
    for name, t in (("q", q), ("k", k), ("v", v), ("alpha", a), ("beta", b)):
        assert torch.isfinite(t.grad).all(), f"non-finite grad wrt {name}"


def _cfg(**kw):
    base = dict(
        vocab_size=64, block_size=64, n_layer=4, n_head=4, n_embd=32,
        dropout=0.0, norm="rms", pos="rope", mlp="swiglu",
    )
    base.update(kw)
    return VariantConfig(**base)


def test_hybrid_routing_places_global_attention_last_in_each_group():
    m = VariantGPT(_cfg(n_layer=8, hybrid_every=4))
    flags = [b.is_linear for b in m.transformer.h]
    assert flags == [True, True, True, False, True, True, True, False]
    linear = sum(flags)
    assert linear == 6 and len(flags) - linear == 2      # the published 3:1 ratio


def test_hybrid_none_is_all_attention():
    m = VariantGPT(_cfg(n_layer=4))
    assert [b.is_linear for b in m.transformer.h] == [False] * 4
    assert not any(isinstance(b.attn, GatedDeltaNet) for b in m.transformer.h)


def test_hybrid_forward_and_backward():
    torch.manual_seed(0)
    m = VariantGPT(_cfg(n_layer=4, hybrid_every=4, gdn_chunk=16))
    idx = torch.randint(0, 64, (2, 64))
    logits, loss = m(idx, targets=idx)
    assert logits.shape == (2, 64, 64)
    loss.backward()
    gdn = m.transformer.h[0].attn
    assert isinstance(gdn, GatedDeltaNet)
    for name, p in gdn.named_parameters():
        assert p.grad is not None, f"no grad reached {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"


def test_gate_init_survives_model_init_weights():
    """REGRESSION: VariantGPT.apply(_init_weights) treats every nn.Linear alike and
    silently overwrote GatedDeltaNet's gate init, starting alpha at sigmoid(0)~0.5 rather
    than ~0.99. That drove the chunk-cumulative decay to 1e-23 and NaN'd training while
    every fp64 unit test still passed. Assert the init a full model actually ends up with,
    not the one the layer sets in isolation."""
    m = VariantGPT(_cfg(n_layer=4, hybrid_every=4, gdn_chunk=16))
    for i, blk in enumerate(m.transformer.h):
        if not blk.is_linear:
            continue
        g = blk.attn
        assert torch.allclose(g.a_proj.bias, torch.full_like(g.a_proj.bias, 4.5)), \
            f"layer {i}: decay-gate bias was clobbered ({g.a_proj.bias[0].item():.3f})"
        assert torch.allclose(g.a_proj.weight, torch.zeros_like(g.a_proj.weight)), \
            f"layer {i}: decay-gate weight was clobbered"
        alpha0 = torch.sigmoid(g.a_proj.bias)[0].item()
        assert alpha0 > 0.98, f"layer {i}: alpha starts at {alpha0:.4f}, want >0.98"
        # And the consequence that actually matters: decay over one chunk stays sane.
        assert alpha0 ** 64 > 1e-2, "cumulative chunk decay underflows at init"


def test_gdn_rejects_kv_cache():
    """A recurrent state is not a KV cache; serving must fail loudly rather than
    silently produce wrong continuations."""
    m = VariantGPT(_cfg(n_layer=4, hybrid_every=4, gdn_chunk=16))
    gdn = m.transformer.h[0].attn
    with pytest.raises(NotImplementedError, match="recurrent state"):
        gdn(torch.randn(1, 16, 32), kv_cache=object())
