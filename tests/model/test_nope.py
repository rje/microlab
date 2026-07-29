"""NoPE (no positional encoding) support in VariantGPT: pos="nope" must carry NO
positional information anywhere (no learned wpe, no rotary application — position is
inferable only from the causal mask; Kazemnejad et al. 2305.19466), while leaving the
existing pos="rope" / pos="learned" paths byte-identical. Cached generation must work
for NoPE (the passkey eval generates through the KV cache)."""

from __future__ import annotations

import pytest
import torch

from microlab.infer.reference.kv_cache import KVCache, generate_cached
from microlab.model.reference.sample import generate
from microlab.model.reference.variants import VariantConfig, VariantGPT


def _cfg(**kw):
    base = dict(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32,
                norm="rms", mlp="swiglu", dropout=0.0)
    base.update(kw)
    return VariantConfig(**base)


def _model(seed=1234, **kw) -> VariantGPT:
    torch.manual_seed(seed)
    return VariantGPT(_cfg(**kw)).eval()


# ---------------------------------------------------------------- rope regression pin

def test_rope_forward_unchanged_golden():
    """Golden values captured from the pre-NoPE variants.py on this machine: the rope
    path must stay numerically identical after the NoPE change."""
    model = _model(pos="rope")
    g = torch.Generator().manual_seed(99)
    x = torch.randint(0, 64, (2, 16), generator=g)
    y = torch.randint(0, 64, (2, 16), generator=g)
    with torch.no_grad():
        logits, loss = model(x, y)
    assert loss.item() == pytest.approx(4.192408084869385, abs=1e-5)
    assert logits.sum().item() == pytest.approx(16.973691940307617, abs=1e-3)
    assert logits[0, 5, 17].item() == pytest.approx(0.2471236139535904, abs=1e-5)
    assert logits[1, 15, 63].item() == pytest.approx(0.07052572071552277, abs=1e-5)


# ------------------------------------------------------------------------ nope basics

def test_nope_has_no_positional_state():
    model = _model(pos="nope")
    assert not hasattr(model.transformer, "wpe")  # no learned positions
    # no rotary tables anywhere (persistent or not)
    for name, _buf in model.named_buffers():
        assert "rope" not in name, f"unexpected rotary buffer {name}"
    # param tree identical to the rope arm's: the A/B compares equal-capacity models
    assert model.num_params() == _model(pos="rope").num_params()
    names_nope = {n for n, _ in model.named_parameters()}
    names_rope = {n for n, _ in _model(pos="rope").named_parameters()}
    assert names_nope == names_rope


def test_nope_attention_is_rope_without_rotation():
    """With the SAME weights, NoPE output == RoPE output when the rotation is forced to
    identity (cos=1, sin=0): NoPE must be exactly 'rope minus the rotation'."""
    nope = _model(seed=7, pos="nope")
    rope = _model(seed=7, pos="rope")  # same seed + same param tree -> same init
    for a, b in zip(nope.parameters(), rope.parameters(), strict=True):
        assert torch.equal(a, b)
    for block in rope.transformer.h:
        block.attn.rope_cos.fill_(1.0)  # apply_rope(x) == x * 1 + rotate_half(x) * 0
        block.attn.rope_sin.fill_(0.0)
    g = torch.Generator().manual_seed(3)
    x = torch.randint(0, 64, (2, 12), generator=g)
    with torch.no_grad():
        ln, _ = nope(x)
        lr, _ = rope(x)
    assert torch.allclose(ln, lr, atol=1e-6)


def test_nope_is_invariant_to_permuting_the_past():
    """In a SINGLE-layer model, causal attention at the last position is a multiset
    function of the preceding tokens when no positional signal exists: permuting the
    past must not change the last-position logits under NoPE. RoPE rotates keys by
    absolute position, so the same permutation must change them there."""
    g = torch.Generator().manual_seed(21)
    x = torch.randint(0, 64, (1, 10), generator=g)
    perm = torch.randperm(9, generator=g)
    x_perm = torch.cat([x[:, perm], x[:, -1:]], dim=1)
    assert not torch.equal(x, x_perm)
    with torch.no_grad():
        nope = _model(pos="nope", n_layer=1)
        rope = _model(pos="rope", n_layer=1)
        last = lambda m, t: m(t)[0][0, -1]  # noqa: E731
        assert torch.allclose(last(nope, x), last(nope, x_perm), atol=1e-5)
        assert not torch.allclose(last(rope, x), last(rope, x_perm), atol=1e-5)


def test_learned_pos_still_builds_wpe_and_runs():
    model = _model(pos="learned", norm="layer", mlp="gelu")
    assert hasattr(model.transformer, "wpe")
    x = torch.randint(0, 64, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, 64) and loss.item() > 0


def test_unknown_pos_raises():
    with pytest.raises(ValueError, match="pos"):
        VariantGPT(_cfg(pos="alibi"))


def test_gqa_with_nope_raises():
    # GQAAttention is rope-only; a nope+GQA config must fail loudly, not silently drop rope
    with pytest.raises(AssertionError, match="RoPE"):
        VariantGPT(_cfg(pos="nope", n_kv_head=1))


# ------------------------------------------------------------------- cached generation

def test_nope_cached_generation_matches_uncached():
    model = _model(pos="nope")
    g = torch.Generator().manual_seed(5)
    idx = torch.randint(0, 64, (1, 8), generator=g)
    out_cached = generate_cached(model, idx.clone(), 6, temperature=0.0)
    out_uncached = generate(model, idx.clone(), 6, temperature=0.0)
    assert torch.equal(out_cached, out_uncached)


def test_nope_cached_single_step_matches_full_forward():
    model = _model(pos="nope")
    cfg = model.config
    g = torch.Generator().manual_seed(11)
    idx = torch.randint(0, 64, (1, 10), generator=g)
    with torch.no_grad():
        full, _ = model(idx)
        cache = KVCache(cfg.n_layer, 1, cfg.n_head, cfg.block_size,
                        cfg.n_embd // cfg.n_head)
        pre, _ = model(idx[:, :-1], kv_cache=cache)
        step, _ = model(idx[:, -1:], kv_cache=cache)
    assert torch.allclose(pre[:, -1], full[:, -2], atol=1e-5)
    assert torch.allclose(step[:, -1], full[:, -1], atol=1e-5)


# ------------------------------------------------- beyond-train-length rebuild (eval path)

def test_nope_rebuilds_with_larger_block_size_and_runs_beyond():
    """The length-generalization eval rebuilds the model with a bigger block_size; for
    NoPE there is no rope cache to extend — the same weights must load strictly and run
    at T beyond the trained window."""
    trained = _model(pos="nope")
    big = VariantGPT(_cfg(pos="nope", block_size=64))
    big.load_state_dict(trained.state_dict())  # strict: identical keys/shapes
    x = torch.randint(0, 64, (1, 48))
    logits, _ = big.eval()(x)
    assert logits.shape == (1, 48, 64)
    # and cached generation works past the trained window too
    out = generate_cached(big, x[:, :20], 8, temperature=0.0)
    assert out.shape == (1, 28)
