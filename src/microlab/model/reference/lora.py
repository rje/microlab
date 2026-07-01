"""Reference LoRA / QLoRA tools (Phase 7). LoRALinear wraps a FROZEN base linear with a
low-rank update y = base(x) + scaling * (x A^T B^T); with B initialized to zero the adapter
is a no-op at init (equals the base), and its merged weights reproduce the adapted layer.
`quantize_dequantize` is a from-scratch absmax quantizer for the QLoRA idea (quantize the
frozen base to low-bit to save VRAM, keep adapters in fp)."""

from __future__ import annotations

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Low-rank adapter over a frozen nn.Linear. Trainable params are only A and B."""

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
        return self.base(x) + self.scaling * (x @ self.lora_A.t() @ self.lora_B.t())

    def merged_weight(self) -> torch.Tensor:
        """The effective weight of an equivalent plain Linear: W + scaling * (B @ A)."""
        return self.base.weight + self.scaling * (self.lora_B @ self.lora_A)

    def merge(self) -> nn.Linear:
        """Return a plain nn.Linear whose output equals this adapter's forward."""
        merged = nn.Linear(
            self.base.in_features, self.base.out_features, bias=self.base.bias is not None
        )
        with torch.no_grad():
            merged.weight.copy_(self.merged_weight())
            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias)
        return merged


def apply_lora_to_gpt(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    targets: tuple[str, ...] = ("c_attn", "c_proj", "c_fc"),
) -> nn.Module:
    """Replace named Linear submodules with LoRALinear and freeze everything except the
    adapters. Returns the same model (mutated)."""
    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name in targets:
                setattr(module, name, LoRALinear(child, rank, alpha))
    for name, p in model.named_parameters():
        p.requires_grad = "lora_" in name
    return model


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def quantize_dequantize(w: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Symmetric absmax quantize then dequantize (the QLoRA idea, simplified). Reconstruction
    error per element is bounded by ~scale/2; more bits -> smaller error."""
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().max() / qmax
    if scale == 0:
        return w.clone()
    q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax)
    return q * scale
