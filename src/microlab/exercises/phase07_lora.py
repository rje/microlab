"""Hand-write exercise (Phase 7): the LoRA adapter math and the QLoRA-style quantizer.

Fill in the ``NotImplementedError`` bodies so ``tests/model/test_student_lora.py`` passes.
Graded against ``microlab.model.reference.lora``. See docs/hand-write/phase7-lora.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Low-rank adapter over a FROZEN nn.Linear. __init__ is given (same params/names as the
    reference so weights transfer); you implement forward + merged_weight."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(rank, base.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))  # zero -> no-op at init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """base(x) plus the scaled low-rank update: scaling * (x @ A^T @ B^T)."""
        raise NotImplementedError("base(x) + scaling * (x @ lora_A.t() @ lora_B.t())")

    def merged_weight(self) -> torch.Tensor:
        """Effective weight of an equivalent plain Linear: base.weight + scaling * (B @ A)."""
        raise NotImplementedError("base.weight + scaling * (lora_B @ lora_A)")


def quantize_dequantize(w: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Symmetric absmax quantize-then-dequantize (the QLoRA idea, simplified):
    scale = max(|w|) / (2^(bits-1) - 1); round w/scale, clamp to the int range, times scale.
    Return w unchanged if scale is 0. More bits -> smaller reconstruction error."""
    raise NotImplementedError("absmax quantize then dequantize")
