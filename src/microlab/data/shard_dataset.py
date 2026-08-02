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

    def sequence_at(self, index: int, block_size: int, seed: int) -> tuple:
        """The (x, y) pair for GLOBAL SEQUENCE `index`, as a pure function of
        (seed, index, block_size). No carried RNG state, no rank, no world size.

        This is what makes a run migratable. `get_batch` below picks its shard from a
        `torch.Generator` whose state is serialised into the checkpoint, so resuming
        reproduces the stream only at the SAME world size and the same accumulation
        layout — move from 1 GPU x accum 16 to 8 GPUs x accum 2 and the data order
        silently changes, which is a different training run wearing the same name.

        Addressing sequences by a global index instead lets any (world_size, grad_accum)
        split the SAME set of sequences per step:

            j = micro_step * world_size + rank

        world_size=1, accum=16 -> j = 0..15;  world_size=8, accum=2 -> j = 0..15.
        Identical global batch, identical gradient, distributed differently.
        """
        # SplitMix-style avalanche: adjacent indices must not give adjacent offsets, or
        # consecutive sequences in a step would overlap in the same shard region.
        h = (seed * 0x9E3779B97F4A7C15 + index * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        h ^= h >> 30
        h = (h * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        h ^= h >> 27
        h = (h * 0x94D049BB133111EB) & ((1 << 64) - 1)
        h ^= h >> 31

        # Shard chosen in proportion to length, so sampling is uniform over TOKENS.
        cum = np.cumsum(self.lengths)
        pick = h % int(cum[-1])
        si = int(np.searchsorted(cum, pick, side="right"))
        arr = self.arrays[si]
        hi = len(arr) - block_size - 1
        if hi <= 0:
            raise ValueError(
                f"shard {si} has {len(arr)} tokens, shorter than block_size+1 "
                f"({block_size + 1}); the mix builder must not emit short shards")
        off = int((h >> 17) % hi)
        x = torch.from_numpy(arr[off:off + block_size].astype(np.int64))
        y = torch.from_numpy(arr[off + 1:off + 1 + block_size].astype(np.int64))
        return x, y

    def get_batch_indexed(self, block_size: int, indices, seed: int, device: str = "cpu"):
        """Stack the sequences at `indices` into one micro-batch."""
        pairs = [self.sequence_at(i, block_size, seed) for i in indices]
        x = torch.stack([p[0] for p in pairs])
        y = torch.stack([p[1] for p in pairs])
        if device.startswith("cuda"):
            return (x.pin_memory().to(device, non_blocking=True),
                    y.pin_memory().to(device, non_blocking=True))
        return x.to(device), y.to(device)

    @staticmethod
    def global_indices(step: int, micro_step: int, rank: int, world_size: int,
                       batch_size: int, seqs_per_step: int) -> list[int]:
        """Global sequence indices this (rank, micro_step) owns at `step`.

        The interleave `micro * world_size + rank` is what keeps the per-step SET of
        sequences invariant when (world_size, grad_accum) changes; a contiguous block
        split would not, because the block boundaries move with world size.
        """
        base = step * seqs_per_step
        out = []
        for b in range(batch_size):
            j = (micro_step * world_size + rank) * batch_size + b
            if j >= seqs_per_step:
                raise ValueError(
                    f"sequence {j} exceeds seqs_per_step={seqs_per_step}; "
                    f"tokens_per_step is not divisible by "
                    f"world_size*batch_size*block_size")
            out.append(base + j)
        return out

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
