"""Peri-LN block layout in VariantGPT (arch-review-2026 finding #9): block_norm="peri"
wraps each sublayer as y = x + Norm(Module(Norm(x))) — standard pre-norm PLUS an output
norm (own learnable scale, init ones) on the module result before the residual add
(Peri-LN, arXiv 2502.02732; the Gemma 2/3 pre+post sandwich). The default
block_norm="pre" must stay byte-identical to the pre-field code (golden pins below were
captured from variants.py BEFORE block_norm existed, on this machine). Unknown values
must raise. The cached-generation, grad-checkpoint, dynamo-tracing, and tied-weights
seams must all be orthogonal to block_norm."""

from __future__ import annotations

import pytest
import torch

from microlab.infer.reference.kv_cache import KVCache, generate_cached
from microlab.model.reference.sample import generate
from microlab.model.reference.variants import RMSNorm, VariantConfig, VariantGPT


def _cfg(**kw):
    base = dict(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32,
                norm="rms", pos="rope", mlp="swiglu", dropout=0.0)
    base.update(kw)
    return VariantConfig(**base)


def _model(seed=4321, **kw) -> VariantGPT:
    torch.manual_seed(seed)
    return VariantGPT(_cfg(**kw)).eval()


def _batch(seed=77, shape=(2, 16)):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 64, shape, generator=g)
    y = torch.randint(0, 64, shape, generator=g)
    return x, y


# ------------------------------------------------------------- pre-LN regression pins

def test_pre_forward_unchanged_golden_rms_rope_swiglu():
    """Golden values captured from the pre-block_norm variants.py on this machine: the
    default layout must stay numerically identical after the Peri-LN change."""
    model = _model()  # block_norm not passed: the default must be the old behavior
    x, y = _batch()
    with torch.no_grad():
        logits, loss = model(x, y)
    assert loss.item() == pytest.approx(4.1402764320373535, abs=1e-5)
    assert logits.sum().item() == pytest.approx(11.358945846557617, abs=1e-3)
    assert logits[0, 5, 17].item() == pytest.approx(-0.04031110927462578, abs=1e-5)
    assert logits[1, 15, 63].item() == pytest.approx(-0.23895953595638275, abs=1e-5)
    assert model.num_params() == 27552


def test_pre_forward_unchanged_golden_layer_learned_gelu():
    model = _model(norm="layer", pos="learned", mlp="gelu")
    x, y = _batch()
    with torch.no_grad():
        logits, loss = model(x, y)
    assert loss.item() == pytest.approx(4.175139904022217, abs=1e-5)
    assert logits.sum().item() == pytest.approx(14.659711837768555, abs=1e-3)
    assert logits[0, 5, 17].item() == pytest.approx(-0.018025578930974007, abs=1e-5)
    assert logits[1, 15, 63].item() == pytest.approx(0.18115347623825073, abs=1e-5)
    assert model.num_params() == 28032


def test_pre_state_dict_has_no_post_norm_keys():
    # Existing checkpoints load strictly: the default layout must not grow any keys.
    for key in _model().state_dict():
        assert "_post" not in key, f"unexpected post-norm key {key} in default layout"


# ----------------------------------------------------------------- peri layout basics

def test_peri_adds_exactly_two_norm_scales_per_block():
    pre, peri = _model(), _model(block_norm="peri")
    extra = set(peri.state_dict()) - set(pre.state_dict())
    assert extra == {f"transformer.h.{i}.ln_{j}_post.weight"
                     for i in range(2) for j in (1, 2)}
    cfg = peri.config
    assert peri.num_params() - pre.num_params() == 2 * cfg.n_layer * cfg.n_embd
    for block in peri.transformer.h:
        assert isinstance(block.ln_1_post, RMSNorm)
        assert isinstance(block.ln_2_post, RMSNorm)
        # standard init: the new output-norm scales start at ones
        assert torch.all(block.ln_1_post.weight == 1.0)
        assert torch.all(block.ln_2_post.weight == 1.0)


def test_peri_post_norm_type_follows_norm_config():
    model = _model(block_norm="peri", norm="layer")
    assert isinstance(model.transformer.h[0].ln_1_post, torch.nn.LayerNorm)


def test_peri_shares_init_stream_with_pre():
    # The extra norms draw no RNG (ones init), so with the same seed every shared param
    # is bit-identical across layouts — the A/B arms differ only in the wiring + scales.
    pre, peri = _model(), _model(block_norm="peri")
    peri_params = dict(peri.named_parameters())
    for name, p in pre.named_parameters():
        assert torch.equal(p, peri_params[name]), name


def test_peri_output_differs_from_pre():
    pre, peri = _model(), _model(block_norm="peri")
    x, _ = _batch()
    with torch.no_grad():
        lp, _ = pre(x)
        lq, _ = peri(x)
    assert not torch.allclose(lp, lq)


def test_peri_block_forward_matches_manual_composition():
    # The wiring, verified against the formulation: y = x + Norm(Module(Norm(x))).
    block = _model(block_norm="peri").transformer.h[0]
    h = torch.randn(2, 8, 32, generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        mid = h + block.ln_1_post(block.attn(block.ln_1(h)))
        want = mid + block.ln_2_post(block.mlp(block.ln_2(mid)))
        got = block(h)
    assert torch.allclose(got, want, atol=1e-6)


def test_unknown_block_norm_raises():
    with pytest.raises(ValueError, match="block_norm"):
        VariantGPT(_cfg(block_norm="post"))
    with pytest.raises(ValueError, match="block_norm"):
        VariantGPT(_cfg(block_norm="sandwich"))


def test_peri_composes_with_other_variant_flags():
    # peri is a block-layout knob, orthogonal to norm/pos/mlp/GQA choices.
    x, y = _batch()
    for kw in (dict(norm="layer", pos="learned", mlp="gelu"),
               dict(n_kv_head=1)):  # GQA (rope-only) with peri
        logits, loss = _model(block_norm="peri", **kw)(x, y)
        assert logits.shape == (2, 16, 64) and loss.item() > 0


# --------------------------------------------------------------- interaction seams

def test_peri_tied_weights_and_strict_state_dict_roundtrip():
    peri = _model(block_norm="peri")
    assert peri.transformer.wte.weight is peri.lm_head.weight
    fresh = _model(seed=1, block_norm="peri")
    fresh.load_state_dict(peri.state_dict())  # strict: identical keys/shapes
    assert fresh.transformer.wte.weight is fresh.lm_head.weight
    x, _ = _batch()
    with torch.no_grad():
        la, _ = peri(x)
        lb, _ = fresh(x)
    assert torch.equal(la, lb)


def test_peri_cached_generation_matches_uncached():
    model = _model(block_norm="peri")
    g = torch.Generator().manual_seed(5)
    idx = torch.randint(0, 64, (1, 8), generator=g)
    out_cached = generate_cached(model, idx.clone(), 6, temperature=0.0)
    out_uncached = generate(model, idx.clone(), 6, temperature=0.0)
    assert torch.equal(out_cached, out_uncached)


def test_peri_cached_single_step_matches_full_forward():
    model = _model(block_norm="peri")
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


def test_peri_grad_checkpoint_matches_plain_backward():
    x, y = _batch()
    grads = {}
    for ckpt in (False, True):
        model = _model(block_norm="peri").train()
        model.grad_checkpoint = ckpt
        _, loss = model(x, y)
        loss.backward()
        grads[ckpt] = {n: p.grad.clone() for n, p in model.named_parameters()}
    assert set(grads[False]) == set(grads[True])
    for name, g in grads[False].items():
        assert torch.allclose(g, grads[True][name], atol=1e-6), name
    # and the new norm scales actually receive gradient signal
    assert grads[False]["transformer.h.0.ln_1_post.weight"].abs().sum() > 0


def test_peri_traces_under_dynamo():
    # torch.compile seam (CPU-cheap: dynamo graph capture with the eager backend;
    # fullgraph=True fails loudly on any graph break the peri branch would introduce).
    model = _model(block_norm="peri")
    x, y = _batch()
    with torch.no_grad():
        eager, _ = model(x, y)
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        traced, _ = compiled(x, y)
    assert torch.allclose(eager, traced, atol=1e-6)
