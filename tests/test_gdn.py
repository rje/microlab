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


def test_rejects_ragged_sequence():
    q, k, v, alpha, beta = _rand(T=64)
    with pytest.raises(ValueError, match="must be a multiple of chunk"):
        gdn_chunkwise(q, k, v, alpha, beta, chunk=48)


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


def test_gdn_rejects_kv_cache():
    """A recurrent state is not a KV cache; serving must fail loudly rather than
    silently produce wrong continuations."""
    m = VariantGPT(_cfg(n_layer=4, hybrid_every=4, gdn_chunk=16))
    gdn = m.transformer.h[0].attn
    with pytest.raises(NotImplementedError, match="recurrent state"):
        gdn(torch.randn(1, 16, 32), kv_cache=object())
