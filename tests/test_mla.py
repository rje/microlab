"""Multi-head Latent Attention — the global layer of the Kimi Linear pattern.

MLA compresses K/V into a low-rank latent, caches only that, and up-projects to per-head
K/V. Because our global layers are NoPE, DeepSeek's decoupled-RoPE split is unnecessary and
omitted — which is also why an earlier arithmetic comparison wrongly rejected MLA: it
priced in 64 rope dims this design does not contain.
"""
import pytest
import torch

from microlab.model.reference.variants import MLAAttention, VariantConfig, VariantGPT


def _cfg(**kw):
    base = dict(vocab_size=512, block_size=128, n_layer=8, n_head=4, n_embd=256,
                dropout=0.0, norm="rms", pos="nope", mlp="swiglu", block_norm="peri",
                hybrid_every=4, global_attn="mla", mla_kv_lora=64, gdn_gate="channel")
    base.update(kw)
    return VariantConfig(**base)


def test_hybrid_routes_mla_to_the_global_layers_only():
    m = VariantGPT(_cfg(n_layer=8))
    kinds = [type(b.attn).__name__ for b in m.transformer.h]
    assert kinds == ["GatedDeltaNet"] * 3 + ["MLAAttention"] + \
                    ["GatedDeltaNet"] * 3 + ["MLAAttention"], kinds


def test_every_head_gets_a_distinct_kv():
    """The entire point of MLA over GQA. If the up-projection collapsed (e.g. a reshape bug
    sharing rows across heads) the model would silently become GQA and still train."""
    torch.manual_seed(0)
    a = MLAAttention(_cfg())
    x = torch.randn(2, 32, 256)
    c = a.kv_a_norm(a.kv_a_proj(x))
    kv = a.kv_b_proj(c).view(2, 32, a.n_head, 2 * a.head_dim)
    k = kv[..., :a.head_dim]
    for i in range(a.n_head):
        for j in range(i + 1, a.n_head):
            assert not torch.allclose(k[:, :, i], k[:, :, j], atol=1e-6), \
                f"heads {i} and {j} share a key — the up-projection collapsed"


def test_causality():
    """Perturbing a later position must not change earlier outputs."""
    torch.manual_seed(0)
    a = MLAAttention(_cfg()).eval()
    x = torch.randn(1, 32, 256)
    base = a(x)
    x2 = x.clone()
    x2[:, 20:] += 5.0
    pert = a(x2)
    assert torch.allclose(base[:, :20], pert[:, :20], atol=1e-5)
    assert not torch.allclose(base[:, 20:], pert[:, 20:], atol=1e-4)


def test_latent_is_the_only_thing_worth_caching():
    """Cache size claim: mla_kv_lora values/token, independent of n_head — that is what
    makes it competitive with GQA(2) while keeping per-head distinctness."""
    for lora, heads in ((64, 4), (64, 8), (128, 4)):
        a = MLAAttention(_cfg(mla_kv_lora=lora, n_head=heads, n_embd=256))
        x = torch.randn(1, 16, 256)
        c = a.kv_a_norm(a.kv_a_proj(x))
        assert c.shape[-1] == lora, "latent width must not depend on head count"


def test_rejects_kv_cache_rather_than_mis_caching():
    a = MLAAttention(_cfg())
    with pytest.raises(NotImplementedError, match="compressed latent"):
        a(torch.randn(1, 16, 256), kv_cache=object())


def test_forward_backward_finite():
    torch.manual_seed(0)
    m = VariantGPT(_cfg())
    idx = torch.randint(0, 512, (2, 128))
    logits, loss = m(idx, targets=idx)
    assert logits.shape == (2, 128, 512)
    loss.backward()
    mla = m.transformer.h[3].attn
    for n, p in mla.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n
