"""Reference groupwise quantization (Phase 6): symmetric absmax round-trip, the shape of
every weight-only inference quant scheme (GPTQ/AWQ add smarter rounding on top)."""

from __future__ import annotations

import torch


def quantize_groupwise(w: torch.Tensor, bits: int = 4, group_size: int = 64) -> torch.Tensor:
    """Quantize-dequantize each `group_size` slice of the input dim independently.
    Returns the dequantized tensor (same shape/dtype) so quality impact is measurable."""
    out_f, in_f = w.shape
    assert in_f % group_size == 0, f"in_features {in_f} not divisible by {group_size}"
    qmax = 2 ** (bits - 1) - 1
    groups = w.view(out_f, in_f // group_size, group_size)
    scale = groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    q = torch.clamp(torch.round(groups / scale), -qmax, qmax)
    return (q * scale).view(out_f, in_f)


@torch.no_grad()
def quantize_model_(model: torch.nn.Module, bits: int = 4, group_size: int = 64):
    """In-place round-trip every Linear weight whose in_features divide group_size."""
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and module.weight.size(1) % group_size == 0:
            module.weight.copy_(quantize_groupwise(module.weight, bits, group_size))
    return model
