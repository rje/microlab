"""Token-batch dataset for LM training: given a 1-D tensor of token ids, sample
contiguous (x, y) blocks where y is x shifted by one. Batches are moved to `device`
(with pinned-memory non-blocking transfer for CUDA — a small GPU-throughput win)."""

from __future__ import annotations

import torch


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert data.dim() == 1 and data.numel() > block_size, (
        "need a 1-D token tensor longer than block_size"
    )
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    if device.startswith("cuda"):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
