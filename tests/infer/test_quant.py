import torch

from microlab.infer.reference.quant import quantize_groupwise, quantize_model_
from microlab.model.reference.variants import VariantConfig, VariantGPT


def test_round_trip_error_small_and_shrinks_with_bits():
    torch.manual_seed(0)
    w = torch.randn(64, 128)
    e8 = (quantize_groupwise(w, bits=8) - w).abs().mean()
    e4 = (quantize_groupwise(w, bits=4) - w).abs().mean()
    assert e8 < e4 < w.abs().mean()  # int8 beats int4 beats garbage


def test_group_scales_are_local():
    w = torch.ones(1, 128)
    w[0, :64] = 100.0  # a huge first group must not wreck the second group's precision
    q = quantize_groupwise(w, bits=4, group_size=64)
    assert (q[0, 64:] - 1.0).abs().max() < 0.2


def test_quantize_model_runs_and_changes_weights():
    torch.manual_seed(0)
    cfg = VariantConfig(vocab_size=64, block_size=32, n_layer=1, n_head=4, n_embd=64,
                        norm="rms", pos="rope", mlp="swiglu")
    m = VariantGPT(cfg)
    before = m.transformer.h[0].attn.c_attn.weight.clone()
    quantize_model_(m, bits=4, group_size=32)
    after = m.transformer.h[0].attn.c_attn.weight
    assert not torch.equal(before, after)
    x = torch.randint(0, 64, (1, 8))
    logits, _ = m(x)
    assert logits.isfinite().all()
