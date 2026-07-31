"""KDA (Kimi Delta Attention) reference vs the fused kernel.

KDA generalises Gated DeltaNet by making the forget gate per-channel rather than one
scalar per head. That distinction is the reason this file exists: our first hybrid used
the scalar gate and we then drew conclusions about the KDA lineage from it — including a
NoPE verdict whose entire hypothesis (the recurrence carries position) depends on gate
capacity.
"""
import pytest
import torch

from microlab.model.reference.variants import _fla_kda, gdn_recurrent, kda_recurrent

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="fused kernel needs CUDA")


def _inputs(B=2, H=4, T=256, D=64, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(B, H, T, D, generator=g, dtype=dtype), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(B, H, T, D, generator=g, dtype=dtype), dim=-1)
    v = torch.randn(B, H, T, D, generator=g, dtype=dtype)
    alpha = 0.90 + 0.10 * torch.rand(B, H, T, D, generator=g, dtype=dtype)   # PER-CHANNEL
    beta = torch.rand(B, H, T, generator=g, dtype=dtype)
    return q, k, v, alpha, beta


def test_kda_reduces_to_gdn_when_the_gate_is_scalar():
    """The load-bearing sanity check: a per-channel gate with identical entries IS the
    scalar gate, so KDA must reproduce Gated DeltaNet exactly. If this fails, one of the
    two references is wrong and every hybrid verdict built on them is suspect."""
    q, k, v, _, beta = _inputs(T=128)
    scalar = 0.9 + 0.1 * torch.rand(q.shape[0], q.shape[1], q.shape[2], dtype=torch.float64)
    vec = scalar.unsqueeze(-1).expand(-1, -1, -1, q.shape[-1]).contiguous()
    got = kda_recurrent(q, k, v, vec, beta)
    ref = gdn_recurrent(q, k, v, scalar, beta)
    assert torch.allclose(got, ref, atol=1e-12, rtol=1e-10), \
        f"max diff {(got-ref).abs().max():.2e}"


def test_per_channel_gate_actually_differs_from_scalar():
    """Guard against a broadcast bug silently collapsing KDA back to GDN — which would
    make the whole upgrade a no-op while every test still passed."""
    q, k, v, alpha, beta = _inputs(T=128)
    per_channel = kda_recurrent(q, k, v, alpha, beta)
    collapsed = kda_recurrent(q, k, v, alpha.mean(-1, keepdim=True).expand_as(alpha), beta)
    assert not torch.allclose(per_channel, collapsed, atol=1e-6), \
        "per-channel gate gave the same answer as its own mean — gate is not being applied"


@cuda
@pytest.mark.parametrize("T", [128, 1024])
def test_fused_kda_matches_reference_within_bf16_floor(T):
    q, k, v, alpha, beta = _inputs(T=T)
    ref = kda_recurrent(q, k, v, alpha, beta)
    scale = ref.abs().max().item()
    floor = (ref.bfloat16().double() - ref).abs().max().item() / scale
    got = _fla_kda(q.cuda().bfloat16(), k.cuda().bfloat16(), v.cuda().bfloat16(),
                   alpha.cuda().float(), beta.cuda().float())
    assert got is not None, "fused KDA unavailable — check the fla install"
    err = (got.double().cpu() - ref).abs().max().item() / scale
    assert err < 5 * floor, f"T={T}: err {err:.2e} vs bf16 floor {floor:.2e}"
    corr = torch.corrcoef(torch.stack([got.double().cpu().flatten(), ref.flatten()]))[0, 1]
    assert corr > 0.999, f"correlation {corr:.6f} — different recurrence, not precision"
