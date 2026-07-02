"""Reference mixture-of-experts primitives (Phase 3): top-k routing, the Switch
load-balancing auxiliary loss, and a token-choice MoE MLP built from SwiGLU experts.
The oracle the owner diffs hand-written routing against."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F

from microlab.model.reference.variants import SwiGLUMLP


def route_topk(router_logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax over ALL experts, take the top-k probs per token, renormalize the kept
    probs to sum to 1. Returns (weights (N,k), expert indices (N,k))."""
    probs = F.softmax(router_logits, dim=-1)
    weights, indices = torch.topk(probs, k, dim=-1)
    return weights / weights.sum(-1, keepdim=True), indices


def load_balance_loss(router_probs: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
    """Switch-Transformer auxiliary loss: E * sum_e f_e * P_e, where f_e is the fraction
    of dispatched (token, slot) assignments that went to expert e and P_e is the mean
    router probability for e. Equals 1.0 under perfectly uniform routing, E on collapse."""
    n_experts = router_probs.size(-1)
    one_hot = F.one_hot(expert_indices, n_experts).float()  # (N, k, E)
    f = one_hot.sum(dim=(0, 1)) / expert_indices.numel()
    p = router_probs.mean(0)
    return n_experts * torch.sum(f * p)


class MoEMLP(nn.Module):
    """Token-choice MoE feed-forward: each token is routed to its top-k SwiGLU experts,
    outputs combined with the renormalized router weights. Returns (y, aux_loss) — add
    `aux_coef * aux_loss` (typical coef ~0.01) to the training loss to keep experts busy."""

    def __init__(self, config, n_experts: int = 4, k: int = 2) -> None:
        super().__init__()
        self.n_experts, self.k = n_experts, k
        self.router = nn.Linear(config.n_embd, n_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLUMLP(config) for _ in range(n_experts))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        flat = x.view(-1, C)
        logits = self.router(flat)
        weights, indices = route_topk(logits, self.k)
        aux = load_balance_loss(F.softmax(logits, dim=-1), indices)
        y = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            for slot in range(self.k):
                mask = indices[:, slot] == e
                if mask.any():
                    y[mask] += weights[mask, slot, None] * expert(flat[mask])
        return y.view(B, T, C), aux
