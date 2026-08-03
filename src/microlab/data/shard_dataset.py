"""Memmapped shard dataset: stream (x, y) blocks from uint16 .bin shards to the GPU.
The scale replacement for the in-memory reference get_batch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class ShardDataset:
    """Token shards as one logical stream.

    Shards may be fetched LAZILY from object storage (`fetch=`). Training reads shards in
    random order and touches one per sequence, so nothing requires all of them present to
    start: on a rented box that takes time-to-first-step from ~9 minutes (US) or ~54
    minutes (Asia) down to the time for a single 400 MB shard. On preemptible hardware that
    restart cost is paid on EVERY re-provision, and it is what otherwise forces the run
    onto expensive US-adjacent hosts.

    `lengths` comes from the MANIFEST, not from the files, so the dataset knows its own
    size — and every sequence index maps to the same shard — before anything is downloaded.
    That is what keeps `sequence_at` a pure function of (seed, index) whether shards are
    local or remote.
    """

    def prefetch(self, indices, block_size: int, seed: int) -> None:
        self._seed = seed
        """Warm the shards `indices` will touch, on background threads.

        Without this, lazy fetch does not save time — it just moves the download inside
        training. A step consumes `tokens_per_step / block_size` sequences (16 for the 1B),
        and random selection means each lands in a different shard, so a blocking fetch
        stalls the step ~16 times. Measured on a rented 4x box: no step completed for 15
        minutes while 47 shards came down serially.

        Because `sequence_at` is a pure function of (seed, index), the shards a FUTURE step
        needs are computable now — that is what makes prefetching exact rather than
        speculative.
        """
        if self.fetch is None:
            return
        want = {self._shard_of(i) for i in indices}
        todo = [i for i in want if i not in self._cache and i not in self._inflight]
        if not todo:
            return
        import threading
        for i in todo:
            self._inflight.add(i)
            threading.Thread(target=self._warm, args=(i,), daemon=True).start()

    def _warm(self, i: int) -> None:
        try:
            self._array(i)
        except Exception as e:                      # noqa: BLE001
            # A prefetch failure must not kill training; the blocking path will retry and
            # raise there, where the error is attributable to a specific sequence.
            print(f"  [prefetch] shard {i} failed: {type(e).__name__}: {e}", flush=True)
        finally:
            self._inflight.discard(i)

    def _shard_of(self, index: int, seed: int = 0) -> int:
        """Which shard sequence `index` lands in — the same arithmetic as sequence_at."""
        h = (self._seed * 0x9E3779B97F4A7C15 + index * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        h ^= h >> 30
        h = (h * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        h ^= h >> 27
        h = (h * 0x94D049BB133111EB) & ((1 << 64) - 1)
        h ^= h >> 31
        cum = np.cumsum(self.lengths)
        return int(np.searchsorted(cum, h % int(cum[-1]), side="right"))

    def __init__(self, data_dir: str, split: str = "train", fetch=None) -> None:
        d = Path(data_dir)
        self.dir = d
        self.split = split
        self.fetch = fetch
        manifest = json.loads((d / f"{split}-manifest.json").read_text())
        shards = manifest["shards"]
        if not shards:
            raise ValueError(f"no shards for split {split!r} in {data_dir}")
        self.files = [d / s["file"] for s in shards]
        # uint16: bytes = 2 * tokens. Taking lengths from the manifest rather than from
        # len(memmap) is what allows a shard to be absent at construction time.
        self.lengths = [int(s["tokens"]) for s in shards]
        self.total_tokens = sum(self.lengths)
        self._cache: dict[int, np.memmap] = {}
        self._inflight: set[int] = set()
        self._seed = 0
        # ONE lock per shard. Without it the main thread and a prefetch thread could fetch
        # the SAME shard concurrently, both writing the same `.part` file: boto3 writes
        # multipart chunks at offsets, so two writers corrupt it, and whichever renames
        # first leaves the other stat-ing a path that no longer exists. Serialising per
        # shard means the second caller waits and then finds it cached.
        import threading
        self._locks: dict[int, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        if fetch is None:
            # Eager: open everything now, and cross-check the manifest against reality.
            # A manifest that disagrees with the files on disk would silently shift every
            # sequence index.
            for i, (p, n) in enumerate(zip(self.files, self.lengths, strict=True)):
                a = np.memmap(p, dtype=np.uint16, mode="r")
                if len(a) != n:
                    raise ValueError(
                        f"{p.name}: manifest says {n:,} tokens, file holds {len(a):,}")
                self._cache[i] = a

    @property
    def arrays(self):
        """Back-compat view. Materialises every shard, so it is only for eager use."""
        return [self._array(i) for i in range(len(self.files))]

    def _lock_for(self, i: int):
        with self._locks_guard:
            import threading
            if i not in self._locks:
                self._locks[i] = threading.Lock()
            return self._locks[i]

    def _array(self, i: int) -> np.memmap:
        a = self._cache.get(i)
        if a is not None:
            return a
        with self._lock_for(i):
            # Re-check: another thread may have completed this shard while we waited.
            a = self._cache.get(i)
            if a is not None:
                return a
            return self._fetch_and_open(i)

    def _fetch_and_open(self, i: int) -> np.memmap:
        p = self.files[i]
        want = self.lengths[i] * 2
        if self.fetch is not None and (not p.exists() or p.stat().st_size != want):
            # Fetch to a temp name and rename, so a shard interrupted mid-download is never
            # visible as a complete one — a truncated shard reads as valid uint16 and would
            # train on short data without complaining.
            p.parent.mkdir(parents=True, exist_ok=True)
            # Per-shard temp name: two DIFFERENT shards may download concurrently, and a
            # shared temp path would have them overwrite each other.
            tmp = p.with_suffix(p.suffix + f".part{i}")
            self.fetch(p.name, tmp)
            if tmp.stat().st_size != want:
                raise ValueError(
                    f"{p.name}: fetched {tmp.stat().st_size:,} bytes, manifest implies "
                    f"{want:,}")
            tmp.replace(p)
        a = np.memmap(p, dtype=np.uint16, mode="r")
        if len(a) != self.lengths[i]:
            raise ValueError(
                f"{p.name}: manifest says {self.lengths[i]:,} tokens, file holds {len(a):,}")
        self._cache[i] = a
        return a

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
        arr = self._array(si)
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
        arr = self._array(si)
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
