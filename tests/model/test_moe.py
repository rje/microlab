"""MoE reference oracle tests: routing math, load-balance loss, end-to-end MoE MLP."""

import torch

from microlab.model.reference.moe import MoEMLP, load_balance_loss, route_topk
from microlab.model.reference.variants import VariantConfig


def test_route_topk_selects_best_and_renormalizes():
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0], [-1.0, 0.0, 3.0, 1.0]])
    weights, idx = route_topk(logits, k=2)
    assert idx.tolist() == [[0, 1], [2, 3]]
    assert torch.allclose(weights.sum(-1), torch.ones(2), atol=1e-6)
    probs = torch.softmax(logits, dim=-1)
    expected0 = probs[0, [0, 1]] / probs[0, [0, 1]].sum()
    assert torch.allclose(weights[0], expected0, atol=1e-6)


def test_load_balance_loss_uniform_is_one():
    # Perfectly uniform routing gives the Switch aux loss its minimum value of 1.0.
    N, E = 64, 4
    probs = torch.full((N, E), 1.0 / E)
    idx = torch.arange(N).remainder(E).unsqueeze(1)  # round-robin dispatch, k=1
    loss = load_balance_loss(probs, idx)
    assert torch.allclose(loss, torch.tensor(1.0), atol=1e-6)


def test_load_balance_loss_collapse_is_e():
    # Total collapse onto one expert scores E (the worst case), penalizing imbalance.
    N, E = 64, 4
    probs = torch.zeros(N, E)
    probs[:, 0] = 1.0
    idx = torch.zeros(N, 1, dtype=torch.long)
    assert torch.allclose(load_balance_loss(probs, idx), torch.tensor(float(E)), atol=1e-6)


def test_moe_mlp_forward_and_backward():
    cfg = VariantConfig(vocab_size=64, block_size=32, n_layer=1, n_head=4, n_embd=32)
    moe = MoEMLP(cfg, n_experts=4, k=2)
    x = torch.randn(2, 8, 32)
    y, aux = moe(x)
    assert y.shape == x.shape and aux.ndim == 0
    (y.mean() + 0.01 * aux).backward()
    assert moe.router.weight.grad is not None
