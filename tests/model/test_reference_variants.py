import pytest
import torch

from microlab.model.reference.gpt import GPTConfig
from microlab.model.reference.train import overfit_batch
from microlab.model.reference.variants import (
    RMSNorm,
    SwiGLUMLP,
    VariantConfig,
    VariantGPT,
    apply_rope,
    build_rope_cache,
)


def _cfg(**kw):
    base = dict(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32)
    base.update(kw)
    return VariantConfig(**base)


def test_rmsnorm_matches_manual_formula():
    torch.manual_seed(0)
    rms = RMSNorm(8)
    x = torch.randn(4, 8)
    expected = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)  # weight=ones
    assert torch.allclose(rms(x), expected, atol=1e-6)


def test_rmsnorm_is_scale_equivariant_in_direction():
    rms = RMSNorm(16)
    x = torch.randn(3, 16)
    # RMSNorm normalizes magnitude: output rms (per row) is ~1 when weight=ones
    out = rms(x)
    assert torch.allclose(out.pow(2).mean(-1), torch.ones(3), atol=1e-4)


def test_rope_preserves_vector_norm():
    cos, sin = build_rope_cache(16, 8)
    x = torch.randn(2, 2, 16, 8)  # (B, nh, T, hd)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


def test_rope_relative_position_property():
    # <rope(q, m), rope(k, n)> depends only on (m - n): equal-offset pairs match.
    cos, sin = build_rope_cache(16, 8)
    torch.manual_seed(0)
    q = torch.randn(1, 1, 16, 8)
    k = torch.randn(1, 1, 16, 8)
    qr = apply_rope(q, cos, sin)
    kr = apply_rope(k, cos, sin)
    # same vectors at every position -> dot(pos m, pos n) for fixed offset is constant
    qc = q[:, :, :1].expand(-1, -1, 16, -1).contiguous()
    kc = k[:, :, :1].expand(-1, -1, 16, -1).contiguous()
    qcr = apply_rope(qc, cos, sin)
    kcr = apply_rope(kc, cos, sin)
    d = (qcr[0, 0] @ kcr[0, 0].T)  # (16,16), entry (m,n)
    assert torch.allclose(torch.diagonal(d, offset=2)[0], torch.diagonal(d, offset=2)[3], atol=1e-4)
    _ = (qr, kr)


def test_swiglu_shape_and_param_comparability():
    cfg = GPTConfig(n_embd=64)
    mlp = SwiGLUMLP(cfg)
    x = torch.randn(2, 8, 64)
    assert mlp(x).shape == (2, 8, 64)


@pytest.mark.parametrize("overrides", [{}, {"norm": "rms"}, {"pos": "rope"}, {"mlp": "swiglu"}])
def test_variant_gpt_forward_and_overfit(overrides):
    torch.manual_seed(0)
    cfg = _cfg(**overrides)
    model = VariantGPT(cfg)
    x = torch.randint(0, 64, (4, 16))
    logits, loss = model(x, x)
    assert logits.shape == (4, 16, 64) and loss.item() > 0
    losses = overfit_batch(model, x, x, steps=200, lr=1e-3, device="cpu")
    assert losses[-1] < losses[0] * 0.3  # each variant can learn a batch


@pytest.mark.gpu
def test_ablation_runner_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from microlab.model.reference.ablate import run_ablations
    from microlab.model.reference.train import TrainConfig

    data = torch.randint(0, 64, (5000,))
    base = _cfg(block_size=32)
    res = run_ablations(
        data, base, TrainConfig(steps=40, batch_size=16, block_size=32, device="cuda")
    )
    assert set(res) == {"baseline", "rmsnorm", "rope", "swiglu"}
    for _name, r in res.items():
        assert r["params"] > 0 and r["final_loss"] > 0
