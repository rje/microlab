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


def write_shards(token_stream: Iterable[int], out_dir: str, split: str = "train",
                 shard_size: int = 100_000_000) -> dict:
    """Write a stream of token ids to uint16 .bin shards + `<split>-manifest.json`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shards: list[dict] = []
    buf: list[int] = []
    total = 0
    idx = 0

    def flush() -> None:
        nonlocal buf, idx
        if not buf:
            return
        arr = np.array(buf, dtype=np.uint16)
        name = f"{split}-{idx:05d}.bin"
        arr.tofile(out / name)
        shards.append({"file": name, "tokens": int(len(arr))})
        idx += 1
        buf = []

    for tid in token_stream:
        buf.append(int(tid))
        total += 1
        if len(buf) >= shard_size:
            flush()
    flush()
    manifest = {"split": split, "shards": shards, "total_tokens": total, "dtype": "uint16"}
    (out / f"{split}-manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
