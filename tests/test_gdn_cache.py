"""Incremental decoding for the GDN/KDA hybrid, and the memory claim it exists to prove.

The correctness bar is the same one the dense KV cache already meets: greedy cached
generation must be token-for-token identical to the uncached path. Without that, the
memory numbers below are measuring a model that produces different text.
"""

import pytest
import torch

from microlab.infer.reference.kv_cache import HybridCache, build_cache, generate_cached
from microlab.model.reference.sample import generate
from microlab.model.reference.variants import (
    VariantConfig,
    VariantGPT,
    gdn_chunkwise,
    gdn_recurrent,
    gdn_step,
)


def _cfg(**kw):
    base = dict(vocab_size=128, block_size=256, n_layer=8, n_head=4, n_embd=64,
                dropout=0.0, norm="rms", pos="rope", mlp="swiglu", block_norm="peri",
                gdn_chunk=16)
    base.update(kw)
    return VariantConfig(**base)


def _rand(B=2, H=3, T=8, Dk=8, Dv=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.nn.functional.normalize(
        torch.randn(B, H, T, Dk, generator=g, dtype=torch.float64), dim=-1)
    k = torch.nn.functional.normalize(
        torch.randn(B, H, T, Dk, generator=g, dtype=torch.float64), dim=-1)
    v = torch.randn(B, H, T, Dv, generator=g, dtype=torch.float64)
    alpha = 0.9 + 0.1 * torch.rand(B, H, T, generator=g, dtype=torch.float64)
    beta = torch.rand(B, H, T, generator=g, dtype=torch.float64)
    return q, k, v, alpha, beta


def test_gdn_step_matches_recurrent():
    """One-token stepping must reproduce the reference recurrence exactly."""
    q, k, v, alpha, beta = _rand(T=12, seed=1)
    ref = gdn_recurrent(q, k, v, alpha, beta)
    B, H, T, Dk = q.shape
    S = torch.zeros(B, H, Dk, v.shape[-1], dtype=torch.float64)
    outs = []
    for t in range(T):
        o, S = gdn_step(q[:, :, t:t+1], k[:, :, t:t+1], v[:, :, t:t+1],
                        alpha[:, :, t:t+1], beta[:, :, t:t+1], S)
        outs.append(o)
    got = torch.cat(outs, dim=2)
    assert torch.allclose(ref, got, atol=1e-10, rtol=1e-8)


def test_prefill_then_step_matches_full_sequence():
    """The real decode pattern: chunkwise prefill carrying state, then single steps."""
    q, k, v, alpha, beta = _rand(T=32, seed=2)
    ref = gdn_recurrent(q, k, v, alpha, beta)
    pre = 16
    y0, S = gdn_chunkwise(q[:, :, :pre], k[:, :, :pre], v[:, :, :pre],
                          alpha[:, :, :pre], beta[:, :, :pre],
                          chunk=16, return_state=True)
    assert torch.allclose(ref[:, :, :pre], y0, atol=1e-10, rtol=1e-8)
    for t in range(pre, 32):
        o, S = gdn_step(q[:, :, t:t+1], k[:, :, t:t+1], v[:, :, t:t+1],
                        alpha[:, :, t:t+1], beta[:, :, t:t+1], S)
        assert torch.allclose(ref[:, :, t:t+1], o, atol=1e-10, rtol=1e-8), f"step {t}"


# Every architecture we can TRAIN must appear here. This gate was previously parameterised
# over `hybrid_every` only, so it kept testing the GQA hybrid after the global layers were
# switched to MLA: green suite, and the shipping architecture had no decode path at all.
# One row per trainable combination is the whole point — if a config can be trained, it
# must be able to generate.
ARCHS = {
    "dense-gqa": dict(hybrid_every=None),
    "hybrid-gqa": dict(hybrid_every=4),
    "hybrid-mla-nope": dict(hybrid_every=4, global_attn="mla", pos="nope",
                            gdn_gate="channel", mla_kv_lora=32, qk_norm=True),
    "dense-mla-nope": dict(hybrid_every=None, global_attn="mla", pos="nope",
                           mla_kv_lora=32),
}


@pytest.mark.parametrize("arch", list(ARCHS), ids=list(ARCHS))
def test_cached_generation_matches_uncached(arch):
    """THE correctness gate. Same guarantee the dense cache already provides; if this
    fails, the cached path is producing different text and any memory saving is
    meaningless."""
    torch.manual_seed(0)
    m = VariantGPT(_cfg(**ARCHS[arch])).eval()
    idx = torch.randint(0, 128, (2, 12))
    slow = generate(m, idx.clone(), max_new_tokens=24, temperature=0.0)
    fast = generate_cached(m, idx.clone(), max_new_tokens=24, temperature=0.0)
    assert torch.equal(slow, fast), (
        f"{arch}: cached and uncached generation diverged at "
        f"position {(slow != fast).float().argmax().item()}"
    )


def test_conv_history_actually_matters():
    """Guard against the tempting shortcut of zero-padding the depthwise conv on every
    decode step. If the history were ignored, cached generation would still run and still
    look plausible — it would just be a different model. This asserts the history is
    non-trivial, so the test above is really exercising it."""
    torch.manual_seed(0)
    m = VariantGPT(_cfg(hybrid_every=4)).eval()
    idx = torch.randint(0, 128, (1, 8))
    cache = build_cache(m, 1, idx.device)
    m(idx, kv_cache=cache)
    hist = [cache.conv_hist(i, 1, 1, 3, torch.float32, idx.device)
            for i in sorted(cache.linear_layers)]
    assert all(h.numel() > 0 for h in hist), "no conv history was retained"
    assert any(h.abs().max() > 0 for h in hist), "conv history is all zeros"


def test_hybrid_cache_allocates_no_kv_for_linear_layers():
    """The memory claim, structurally: linear layers must have NO K/V buffer at all."""
    m = VariantGPT(_cfg(n_layer=8, hybrid_every=4))
    cache = build_cache(m, 1, "cpu")
    assert isinstance(cache, HybridCache)
    assert cache.linear_layers == {0, 1, 2, 4, 5, 6}
    for i in sorted(cache.linear_layers):
        assert cache.k[i] is None and cache.v[i] is None, f"layer {i} allocated K/V"
    for i in (3, 7):
        assert cache.k[i] is not None, f"global layer {i} missing K/V"
    with pytest.raises(AssertionError, match="no K/V buffer"):
        cache.append(0, torch.zeros(1, 4, 1, 16), torch.zeros(1, 4, 1, 16))


def test_hybrid_cache_rejects_linear_last_layer():
    """seq_len is advanced by the last layer, so a linear final layer must fail loudly."""
    with pytest.raises(ValueError, match="final layer to be global"):
        HybridCache(n_layer=4, batch_size=1, n_kv_head=4, capacity=16, head_dim=8,
                    linear_layers={3}, n_head=4)


def test_gdn_state_is_constant_in_context_length():
    """The architectural claim: KV grows with context, the recurrent state does not."""
    torch.manual_seed(0)
    m = VariantGPT(_cfg(n_layer=8, hybrid_every=4, block_size=256)).eval()
    sizes = {}
    for T in (16, 64, 192):
        cache = build_cache(m, 1, "cpu")
        m(torch.randint(0, 128, (1, T)), kv_cache=cache)
        sizes[T] = (cache.kv_bytes(), cache.state_bytes())
    state = {s for _, s in sizes.values()}
    assert len(state) == 1, f"recurrent state grew with context: {sizes}"
    kv = [sizes[T][0] for T in (16, 64, 192)]
    assert kv[0] < kv[1] < kv[2], f"KV did not grow with context: {kv}"
