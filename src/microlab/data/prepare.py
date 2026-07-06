"""Real-scale data pipeline (Phase-1): tokenize a corpus into uint16 .bin shards + a JSON
manifest for memmapped streaming, with a train/val split and eval-contamination stripping.
uint16 requires vocab_size <= 65536 (our 32k fits)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np


def strip_contamination(texts: Iterable[str], eval_prompts: list[str]) -> Iterator[str]:
    """Drop any document containing a Phase-0 eval prompt verbatim (keeps eval honest)."""
    bad = [p for p in eval_prompts if p]
    for t in texts:
        if not any(p in t for p in bad):
            yield t


def write_shards(chunks: Iterable[np.ndarray], out_dir: str, split: str = "train",
                 shard_size: int = 100_000_000) -> dict:
    """Write a stream of uint16 token ARRAYS to `<split>-NNNNN.bin` shards + a manifest. Consuming
    arrays (not individual ids) keeps the whole pipeline off the per-token Python path: the
    tokenizer's encode_batch produces arrays and numpy writes them in bulk — the difference
    between ~1M and >10M tokens/sec at scale. Shards are exactly `shard_size` tokens (the last is
    the remainder). Accepts an iterable of int, too, for small/reference use."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shards: list[dict] = []
    pending: list[np.ndarray] = []  # accumulated arrays not yet at a shard boundary
    pending_len = 0
    total = 0
    idx = 0

    def _write(arr: np.ndarray) -> None:
        nonlocal idx
        name = f"{split}-{idx:05d}.bin"
        arr.astype(np.uint16, copy=False).tofile(out / name)
        shards.append({"file": name, "tokens": int(len(arr))})
        idx += 1

    def drain(final: bool) -> None:
        """Emit full `shard_size` shards from the pending buffer; on final, the remainder too."""
        nonlocal pending, pending_len
        if pending_len == 0:
            return
        data = np.concatenate(pending) if len(pending) > 1 else pending[0]
        off = 0
        while len(data) - off >= shard_size:
            _write(data[off:off + shard_size])
            off += shard_size
        rem = data[off:]
        if final and len(rem):
            _write(rem)
            rem = rem[:0]
        pending = [rem] if len(rem) else []
        pending_len = len(rem)

    for chunk in chunks:
        arr = np.asarray(chunk, dtype=np.uint16).reshape(-1)
        pending.append(arr)
        pending_len += len(arr)
        total += len(arr)
        if pending_len >= shard_size:
            drain(final=False)
    drain(final=True)
    manifest = {"split": split, "shards": shards, "total_tokens": int(total), "dtype": "uint16"}
    (out / f"{split}-manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def batched_token_chunks(tok, docs: Iterable[str], eot: int,
                         batch_docs: int = 1024) -> Iterator[np.ndarray]:
    """Tokenize `docs` in batches of `batch_docs` (tok.encode_batch runs the Rust tokenizer
    across all cores), yielding one uint16 array per batch with an EOT after each document. The
    scale replacement for a per-document ``yield from tok.encode(text)`` generator."""
    eot_arr = np.array([eot], dtype=np.uint16)

    def _flush(batch: list[str]) -> np.ndarray:
        parts: list[np.ndarray] = []
        for ids in tok.encode_batch(batch):
            parts.append(np.asarray(ids, dtype=np.uint16))
            parts.append(eot_arr)
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.uint16)

    batch: list[str] = []
    for text in docs:
        batch.append(text)
        if len(batch) >= batch_docs:
            yield _flush(batch)
            batch = []
    if batch:
        yield _flush(batch)


def take_tokens(chunks: Iterator[np.ndarray], budget: int,
                carry: list[np.ndarray]) -> Iterator[np.ndarray]:
    """Yield arrays totaling exactly `budget` tokens from a SHARED chunk iterator, splitting the
    boundary chunk and stashing its remainder in `carry` so the next call resumes with no loss
    or overlap — the array-stream equivalent of itertools.islice for the train/val split."""
    total = 0
    while total < budget:
        chunk = carry.pop() if carry else next(chunks, None)
        if chunk is None:
            return
        if total + len(chunk) <= budget:
            total += len(chunk)
            yield chunk
        else:
            cut = budget - total
            yield chunk[:cut]
            carry.append(chunk[cut:])
            return
