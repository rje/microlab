"""Spec + validation for the hand-written Phase-3 architecture primitives.

Implement ``microlab.exercises.phase03_variants`` until these pass. Each test diffs your
work against the reference oracle in ``microlab.model.reference.variants``.
"""

import pytest
import torch

from microlab.exercises.phase03_variants import (
    GQAAttention,
    RMSNorm,
    SwiGLUMLP,
    apply_rope,
    load_balance_loss,
    route_topk,
)
from microlab.model.reference import moe as ref_moe
from microlab.model.reference.gpt import GPTConfig
from microlab.model.reference.variants import GQAAttention as RefGQA
from microlab.model.reference.variants import RMSNorm as RefRMSNorm
from microlab.model.reference.variants import SwiGLUMLP as RefSwiGLU
from microlab.model.reference.variants import VariantConfig, build_rope_cache
from microlab.model.reference.variants import apply_rope as ref_apply_rope


def test_rmsnorm_matches_reference():
    torch.manual_seed(0)
    ref, stu = RefRMSNorm(16), RMSNorm(16)
    stu.load_state_dict(ref.state_dict())
    x = torch.randn(4, 8, 16)
    assert torch.allclose(stu(x), ref(x), atol=1e-6)


def test_rmsnorm_normalizes_magnitude():
    stu = RMSNorm(16)
    x = torch.randn(3, 16)
    assert torch.allclose(stu(x).pow(2).mean(-1), torch.ones(3), atol=1e-4)


def test_apply_rope_matches_reference():
    cos, sin = build_rope_cache(16, 8)
    torch.manual_seed(0)
    x = torch.randn(2, 2, 16, 8)
    assert torch.allclose(apply_rope(x, cos, sin), ref_apply_rope(x, cos, sin), atol=1e-6)


def test_apply_rope_preserves_norm():
    cos, sin = build_rope_cache(16, 8)
    x = torch.randn(2, 2, 16, 8)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


def test_swiglu_matches_reference():
    torch.manual_seed(0)
    cfg = GPTConfig(n_embd=64)
    ref, stu = RefSwiGLU(cfg).eval(), SwiGLUMLP(cfg).eval()
    stu.load_state_dict(ref.state_dict())
    x = torch.randn(2, 8, 64)
    assert torch.allclose(stu(x), ref(x), atol=1e-5)


@pytest.mark.parametrize("n_kv", [1, 3, 6])
def test_gqa_matches_reference(n_kv):
    torch.manual_seed(0)
    cfg = VariantConfig(vocab_size=64, block_size=32, n_layer=2, n_head=6, n_embd=48,
                        norm="rms", pos="rope", mlp="swiglu", n_kv_head=n_kv)
    ref, stu = RefGQA(cfg).eval(), GQAAttention(cfg).eval()
    stu.load_state_dict(ref.state_dict())
    x = torch.randn(2, 16, 48)
    assert torch.allclose(stu(x), ref(x), atol=1e-5)


def test_route_topk_matches_reference():
    torch.manual_seed(0)
    logits = torch.randn(32, 8)
    w_s, i_s = route_topk(logits, k=2)
    w_r, i_r = ref_moe.route_topk(logits, k=2)
    assert torch.equal(i_s, i_r) and torch.allclose(w_s, w_r, atol=1e-6)


def test_load_balance_loss_matches_reference():
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(64, 4), dim=-1)
    idx = torch.randint(0, 4, (64, 2))
    assert torch.allclose(
        load_balance_loss(probs, idx), ref_moe.load_balance_loss(probs, idx), atol=1e-6
    )


pytestmark = pytest.mark.exercise
