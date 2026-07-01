"""Hand-write exercise (Phase 3): implement the three architecture primitives —
RMSNorm, rotary position embeddings (RoPE), and a SwiGLU MLP.

Fill in the ``NotImplementedError`` bodies so ``tests/model/test_student_variants.py``
passes. The differential tests copy the reference oracle's weights into your modules
(or call your pure ``apply_rope`` with the same inputs) and compare outputs, so your
math must match `microlab.model.reference.variants`. See docs/hand-write/phase3-ablations.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from microlab.model.reference.gpt import GPTConfig


class RMSNorm(nn.Module):
    """Root-mean-square layer norm. Same parameter (``weight``) as the reference."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "implement RMSNorm: divide x by sqrt(mean(x^2, last dim) + eps), then scale "
            "by self.weight (no mean-subtraction, no bias)"
        )


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to x of shape (B, n_head, T, head_dim).
    cos/sin are the precomputed tables of shape (T, head_dim//2) from
    ``microlab.model.reference.variants.build_rope_cache``. Use the rotate-half
    convention: split x into halves (x1, x2); the rotation is
    ``x * cos_full + rotate_half(x) * sin_full`` where ``rotate_half(x) = cat(-x2, x1)``
    and cos/sin are duplicated to full head_dim with ``cat((t, t), dim=-1)``."""
    raise NotImplementedError("implement rotary position embedding application")


class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward: ``w2(silu(w1 x) * w3 x)``. Same parameters/names (w1, w3, w2)
    and hidden dim as the reference so its weights load straight in."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        hidden = int(8 / 3 * config.n_embd)
        hidden = ((hidden + 7) // 8) * 8
        self.w1 = nn.Linear(config.n_embd, hidden, bias=False)
        self.w3 = nn.Linear(config.n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "implement SwiGLU: self.w2(silu(self.w1(x)) * self.w3(x)), then dropout"
        )
