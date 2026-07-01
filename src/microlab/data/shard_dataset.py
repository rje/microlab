"""Memmapped shard dataset: stream (x, y) blocks from uint16 .bin shards to the GPU.
The scale replacement for the in-memory reference get_batch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class ShardDataset:
    def __init__(self, data_dir: str, split: str = "train") -> None:
        d = Path(data_dir)
        manifest = json.loads((d / f"{split}-manifest.json").read_text())
        self.arrays = [np.memmap(d / s["file"], dtype=np.uint16, mode="r")
                       for s in manifest["shards"]]
        self.lengths = [len(a) for a in self.arrays]
        self.total_tokens = sum(self.lengths)
        if not self.arrays:
            raise ValueError(f"no shards for split {split!r} in {data_dir}")

    def get_batch(self, block_size: int, batch_size: int, device: str = "cpu",
                  generator: torch.Generator | None = None):
        # pick a random shard (weighted by length), then random offsets within it
        weights = torch.tensor(self.lengths, dtype=torch.float)
        si = int(torch.multinomial(weights, 1, generator=generator).item())
        arr = self.arrays[si]
        hi = len(arr) - block_size - 1
        assert hi > 0, "shard shorter than block_size+1"
        ix = torch.randint(hi, (batch_size,), generator=generator)
        x = torch.stack([torch.from_numpy(arr[i:i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack(
            [torch.from_numpy(arr[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix]
        )
        if device.startswith("cuda"):
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y
