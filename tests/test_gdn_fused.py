"""The fused Triton kernel must compute the same recurrence as our reference.

This is the gate that lets `fla` be the training kernel while `gdn_recurrent` stays the
oracle. It is the same bargain already struck for attention: we call SDPA (FlashAttention)
rather than writing our own, and verify behaviour against a definition we do control.
"""
import pytest
import torch

from microlab.model.reference.variants import _fla_gdn, gdn_recurrent

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="fused kernel needs CUDA")


@cuda
@pytest.mark.parametrize("T", [128, 1024])
def test_fused_matches_reference_within_bf16_floor(T):
    """The kernel computes in bf16, so it cannot beat the bf16 representation floor —
    but it must not be materially worse than it. Comparing against that floor rather
    than a fixed tolerance is what distinguishes 'precision' from 'wrong algorithm'."""
    torch.manual_seed(0)
    B, H, D = 2, 4, 64
    q = torch.nn.functional.normalize(torch.randn(B, H, T, D, dtype=torch.float64), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(B, H, T, D, dtype=torch.float64), dim=-1)
    v = torch.randn(B, H, T, D, dtype=torch.float64)
    alpha = 0.9 + 0.1 * torch.rand(B, H, T, dtype=torch.float64)
    beta = torch.rand(B, H, T, dtype=torch.float64)

    ref = gdn_recurrent(q, k, v, alpha, beta)
    scale = ref.abs().max().item()
    floor = (ref.bfloat16().double() - ref).abs().max().item() / scale

    got = _fla_gdn(*(x.cuda().bfloat16() for x in (q, k, v)),
                   alpha.cuda().float(), beta.cuda().float())
    assert got is not None, "fused path unavailable on CUDA — check the fla install"
    err = (got.double().cpu() - ref).abs().max().item() / scale
    assert err < 3 * floor, f"T={T}: err {err:.2e} vs bf16 floor {floor:.2e}"
    corr = torch.corrcoef(torch.stack([got.double().cpu().flatten(), ref.flatten()]))[0, 1]
    assert corr > 0.9999, f"correlation {corr:.6f} — different recurrence, not precision"


@cuda
def test_fused_declines_on_float64_so_the_oracle_stays_exact():
    q = torch.randn(1, 2, 64, 32, dtype=torch.float64, device="cuda")
    a = torch.rand(1, 2, 64, dtype=torch.float64, device="cuda")
    assert _fla_gdn(q, q, q, a, a) is None


def test_fused_declines_on_cpu():
    q = torch.randn(1, 2, 64, 32)
    a = torch.rand(1, 2, 64)
    assert _fla_gdn(q, q, q, a, a) is None
