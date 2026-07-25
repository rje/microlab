"""VariantConfig.rope_base plumbing: the knob must actually reach build_rope_cache in
both RoPE attention modules, and the default must reproduce the pre-knob behavior
bit-identically (existing checkpoints/runs depend on it)."""

import torch

from microlab.model.reference.variants import (
    GQAAttention,
    RoPECausalSelfAttention,
    VariantConfig,
    build_rope_cache,
)


def _cfg(**kw):
    base = dict(vocab_size=64, block_size=32, n_layer=2, n_head=4, n_embd=32,
                norm="rms", pos="rope", mlp="swiglu")
    base.update(kw)
    return VariantConfig(**base)


def test_default_rope_base_matches_old_cache():
    # Pre-knob behavior: build_rope_cache(base=10000.0). Default config must be identical.
    attn = RoPECausalSelfAttention(_cfg())
    cos, sin = build_rope_cache(32, 8, base=10000.0)
    assert torch.equal(attn.rope_cos, cos)
    assert torch.equal(attn.rope_sin, sin)


def test_rope_base_reaches_mha_cache():
    attn = RoPECausalSelfAttention(_cfg(rope_base=500000.0))
    cos, sin = build_rope_cache(32, 8, base=500000.0)
    assert torch.equal(attn.rope_cos, cos)
    assert torch.equal(attn.rope_sin, sin)
    default = RoPECausalSelfAttention(_cfg())
    assert not torch.equal(attn.rope_cos, default.rope_cos)


def test_rope_base_reaches_gqa_cache():
    attn = GQAAttention(_cfg(n_kv_head=2, rope_base=500000.0))
    cos, sin = build_rope_cache(32, 8, base=500000.0)
    assert torch.equal(attn.rope_cos, cos)
    assert torch.equal(attn.rope_sin, sin)
