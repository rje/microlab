"""GQA reference oracle tests: shapes, MHA-equivalence, and checkpoint compatibility."""

import pytest
import torch

from microlab.model.reference.variants import (
    GQAAttention,
    RoPECausalSelfAttention,
    VariantConfig,
    VariantGPT,
)


def _cfg(n_kv_head=None):
    return VariantConfig(
        vocab_size=64, block_size=32, n_layer=2, n_head=6, n_embd=48,
        norm="rms", pos="rope", mlp="swiglu", n_kv_head=n_kv_head,
    )


@pytest.mark.parametrize("n_kv", [1, 2, 3, 6])
def test_gqa_output_shape(n_kv):
    attn = GQAAttention(_cfg(n_kv_head=n_kv)).eval()
    x = torch.randn(2, 16, 48)
    assert attn(x).shape == (2, 16, 48)


def test_gqa_with_all_heads_equals_mha():
    # n_kv_head == n_head must reproduce the fused-projection MHA exactly when the
    # fused c_attn weights are sliced into q_proj / kv_proj.
    torch.manual_seed(0)
    cfg = _cfg(n_kv_head=6)
    mha = RoPECausalSelfAttention(cfg).eval()
    gqa = GQAAttention(cfg).eval()
    C = cfg.n_embd
    with torch.no_grad():
        gqa.q_proj.weight.copy_(mha.c_attn.weight[:C])
        gqa.kv_proj.weight.copy_(mha.c_attn.weight[C:])
        gqa.c_proj.weight.copy_(mha.c_proj.weight)
        # config.bias defaults to True, so the fused c_attn/c_proj carry biases; slice
        # them the same way as the weights or the two paths diverge by the bias term.
        gqa.q_proj.bias.copy_(mha.c_attn.bias[:C])
        gqa.kv_proj.bias.copy_(mha.c_attn.bias[C:])
        gqa.c_proj.bias.copy_(mha.c_proj.bias)
    x = torch.randn(2, 16, C)
    assert torch.allclose(gqa(x), mha(x), atol=1e-5)


def test_gqa_param_savings():
    full = sum(p.numel() for p in GQAAttention(_cfg(n_kv_head=6)).parameters())
    mqa = sum(p.numel() for p in GQAAttention(_cfg(n_kv_head=1)).parameters())
    assert mqa < full


def test_gqa_causality():
    attn = GQAAttention(_cfg(n_kv_head=2)).eval()
    x = torch.randn(1, 16, 48)
    y1 = attn(x)
    x2 = x.clone()
    x2[0, 10:] = 0.0  # changing the future...
    y2 = attn(x2)
    assert torch.allclose(y1[0, :10], y2[0, :10], atol=1e-6)  # ...can't change the past


def test_default_none_keeps_variantgpt_identical():
    # Checkpoint/live-run safety: n_kv_head=None must be bit-for-bit the old model.
    torch.manual_seed(0)
    a = VariantGPT(_cfg(n_kv_head=None))
    torch.manual_seed(0)
    b = VariantGPT(VariantConfig(vocab_size=64, block_size=32, n_layer=2, n_head=6,
                                 n_embd=48, norm="rms", pos="rope", mlp="swiglu"))
    x = torch.randint(0, 64, (2, 16))
    la, _ = a(x)
    lb, _ = b(x)
    assert torch.equal(la, lb)
    assert list(a.state_dict().keys()) == list(b.state_dict().keys())


def test_variantgpt_trains_with_gqa():
    m = VariantGPT(_cfg(n_kv_head=2))
    x = torch.randint(0, 64, (2, 16))
    _, loss = m(x, x)
    loss.backward()
    assert loss.isfinite()
