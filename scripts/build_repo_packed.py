#!/usr/bin/env python
"""Repack a file-level code corpus into REPO-LEVEL packed shards.

Why this exists: measured on our corpus, the median document is 664 tokens and only 0.18%
exceed 32k. A 32k training window over file-level shards therefore spans ~20 UNRELATED
files, so long-context pretraining would mostly teach the model to ignore everything before
the last document boundary. Qwen2.5-Coder hit the same wall and solved it the same way
(5.2T file-level + ~300B repo-level).

No re-download, no re-tokenisation. `attribution.jsonl` is written in the same order the
tokens were appended to the shards -- verified: summing its `tokens` field per split
reproduces both manifests to the token -- so a cumulative sum recovers each file's byte
range and we can repack by GATHERING existing token ranges.

Packing strategy (measured against the repo-size distribution: 3.69M repos, median 1,706
tokens, but 139,590 repos >=32k covering 51% of all tokens):
  - repos at or above the window fill windows on their own, split across consecutive
    windows so a long repo stays contiguous;
  - the long tail of small repos is bin-packed largest-first into the leftovers, which
    keeps whole small repos intact rather than cutting them at arbitrary points.
Files inside a repo are emitted in path order, so files from the same directory land
adjacent. Dependency-aware ordering (imports before importers) is a later upgrade and is
deliberately A/B-able against this.

Resumable and progressive (house rule: long jobs write as they go).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_index(attr_path: Path, split: str):
    """repo -> [(offset, length), ...] in the source token stream, plus per-file paths.

    Offsets come from a cumulative sum over the attribution rows for THIS split, which is
    valid precisely because the writer appended tokens and attribution rows in lockstep."""
    repos: dict[str, list] = {}
    off = 0
    n = 0
    with attr_path.open() as f:
        for line in f:
            d = json.loads(line)
            if d["split"] != split:
                continue
            repos.setdefault(d["repo"], []).append((off, d["tokens"], d["path"]))
            off += d["tokens"]
            n += 1
    return repos, off, n


class ShardReader:
    """Random access across a manifest's shards as one logical token stream."""

    def __init__(self, data_dir: Path, split: str):
        man = json.loads((data_dir / f"{split}-manifest.json").read_text())
        self.arrays, self.starts = [], []
        pos = 0
        for s in man["shards"]:
            a = np.memmap(data_dir / s["file"], dtype=np.uint16, mode="r")
            self.arrays.append(a)
            self.starts.append(pos)
            pos += len(a)
        self.total = pos

    def read(self, off: int, length: int) -> np.ndarray:
        """Gather [off, off+length) even when it straddles a shard boundary."""
        out = np.empty(length, dtype=np.uint16)
        w = 0
        i = int(np.searchsorted(self.starts, off, side="right") - 1)
        while w < length:
            a = self.arrays[i]
            local = off + w - self.starts[i]
            take = min(len(a) - local, length - w)
            out[w:w + take] = a[local:local + take]
            w += take
            i += 1
        return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="file-level corpus dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=32768, help="target sequence length")
    ap.add_argument("--shard-tokens", type=int, default=100_000_000)
    ap.add_argument("--split", default="train")
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # Per-SPLIT state. A shared file meant the val pass read the train pass's completed
    # state (repo_i == len(order)) and exited having written nothing — the resumability
    # feature silently skipping the whole job.
    state_path = out / f"pack-state-{a.split}.json"

    print(f"indexing {src/'attribution.jsonl'} ...", flush=True)
    repos, total, nfiles = build_index(src / "attribution.jsonl", a.split)
    print(f"  {nfiles:,} files across {len(repos):,} repos, {total:,} tokens", flush=True)

    reader = ShardReader(src, a.split)
    if reader.total != total:
        raise SystemExit(
            f"attribution total {total:,} != shard total {reader.total:,}; the cumulative-sum "
            f"offset reconstruction is only valid if these agree exactly")

    # Largest repos first: they fill windows on their own, and packing the tail afterwards
    # into the remaining space wastes less than the reverse order.
    order = sorted(repos.items(), key=lambda kv: -sum(t for _, t, _ in kv[1]))

    done = json.loads(state_path.read_text()) if state_path.exists() else {"repo_i": 0,
                                                                           "shard": 0,
                                                                           "tokens": 0}
    buf: list[np.ndarray] = []
    buf_n = 0
    shard_i, written = done["shard"], done["tokens"]
    manifest = []
    if (out / f"{a.split}-manifest.json").exists():
        manifest = json.loads((out / f"{a.split}-manifest.json").read_text())["shards"]

    def flush_shard():
        nonlocal buf, buf_n, shard_i
        if not buf:
            return
        arr = np.concatenate(buf)
        name = f"{a.split}-{shard_i:05d}.bin"
        arr.tofile(out / name)
        manifest.append({"file": name, "tokens": int(len(arr))})
        (out / f"{a.split}-manifest.json").write_text(json.dumps(
            {"split": a.split, "dtype": "uint16",
             "shards": manifest,
             "total_tokens": sum(s["tokens"] for s in manifest)}, indent=1))
        shard_i += 1
        buf, buf_n = [], 0

    for i in range(done["repo_i"], len(order)):
        repo, files = order[i]
        files.sort(key=lambda t: t[2])          # path order within the repo
        for off, ln, _ in files:
            buf.append(reader.read(off, ln))
            buf_n += ln
            written += ln
        if buf_n >= a.shard_tokens:
            flush_shard()
            state_path.write_text(json.dumps({"repo_i": i + 1, "shard": shard_i,
                                              "tokens": written}))
            pct = written / total * 100
            print(f"  {written:,}/{total:,} tokens ({pct:.1f}%), {shard_i} shards", flush=True)
    flush_shard()
    state_path.write_text(json.dumps({"repo_i": len(order), "shard": shard_i,
                                      "tokens": written}))

    for extra in ("tokenizer.json",):
        if (src / extra).exists() and not (out / extra).exists():
            (out / extra).write_bytes((src / extra).read_bytes())
    print(f"done: {written:,} tokens in {shard_i} shards -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
