#!/usr/bin/env python
"""Build the pretraining mix from the six built slices, with FIM applied to code.

    python scripts/build_mix.py --out data/shards/mix-v1 \\
        --tokenizer data/tokenizers/code-49k-fim.json --target-tokens 21000000000

WHY A BUILDER AND NOT A LOADER-SIDE WEIGHTING: the mix is part of the experiment. Baking
it into shards makes the corpus a fixed, hashable artifact that a rented 8xH100 job reads
without needing our sampling code, and makes "what did this model train on" answerable
from the manifest alone.

MIXING IS PER-DOCUMENT, NOT BLOCKED. Slices are interleaved by weighted draw for every
document, so any window of the corpus has roughly the target composition. Concatenating
slices instead would give the model 14B tokens of pure code followed by 3B of pure web,
which is a curriculum nobody chose.

FIM (0.5 PSM) IS APPLIED TO CODE SLICES ONLY. Infilling is a code-editing capability;
spending it on arXiv prose would train the model to reorder paragraphs. See
microlab.data.fim for the transform and its round-trip guarantee.

SLICE COUNTS ARE ASSERTED UP FRONT. The commits slice once ingested nothing and left an
empty directory that looked like success (the ingest died on a loader-script error and the
wrapper's `|| echo` went to a log nobody read). Any slice short of its requirement raises
here, naming the slice, rather than producing a quietly mis-proportioned corpus.

Resumable and progressive (house rule: long jobs write as they go).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microlab.data.fim import FIMConfig, fim_transform, split_documents  # noqa: E402

# name -> (directory, share of the mix, is_code)
# Shares reflect the plan's 65.5/15/10/5/2.5/2 with one measured correction: CommitPackFT
# is finite and yielded 331.9M tokens, so commits lands at 1.2% and the missing 0.8% goes
# to code rather than being padded with repeats of a small slice.
SLICES = {
    "code":     ("data/shards/code-repo-32k", 0.663, True),
    "web":      ("data/shards/web-49k",       0.150, False),
    "math":     ("data/shards/math-49k",      0.100, False),
    "markdown": ("data/shards/markdown-49k",  0.050, False),
    "arxiv":    ("data/shards/arxiv-49k",     0.025, False),
    "commits":  ("data/shards/commits-49k",   0.012, True),
}


class SliceReader:
    """Sequential document reader over one slice's shards, with a resumable cursor."""

    def __init__(self, name: str, data_dir: Path, split: str, eot: int) -> None:
        self.name = name
        man = json.loads((data_dir / f"{split}-manifest.json").read_text())
        self.files = [data_dir / s["file"] for s in man["shards"]]
        self.total = man["total_tokens"]
        self.eot = eot
        self.shard_i = 0
        self.docs: list[np.ndarray] = []
        self.doc_i = 0

    def state(self) -> dict:
        return {"shard_i": self.shard_i, "doc_i": self.doc_i}

    def restore(self, st: dict) -> None:
        self.shard_i, self.doc_i = st["shard_i"], st["doc_i"]
        self.docs = []

    def next_doc(self) -> np.ndarray | None:
        while self.doc_i >= len(self.docs):
            if self.shard_i >= len(self.files):
                return None                       # slice exhausted
            arr = np.fromfile(self.files[self.shard_i], dtype=np.uint16)
            self.docs = split_documents(arr, self.eot)
            self.shard_i += 1
            if self.doc_i >= len(self.docs):      # resumed past this shard
                self.doc_i -= len(self.docs)
                self.docs = []
        d = self.docs[self.doc_i]
        self.doc_i += 1
        return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", required=True, help="MUST carry the FIM sentinels")
    ap.add_argument("--target-tokens", type=int, required=True)
    ap.add_argument("--val-tokens", type=int, default=50_000_000)
    ap.add_argument("--shard-tokens", type=int, default=200_000_000)
    ap.add_argument("--fim-rate", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--split", default="train")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(a.tokenizer)
    eot = tok.token_to_id("<|endoftext|>")
    fim = FIMConfig(tok)

    # --- gate: every slice must exist and hold enough tokens for its share -------------
    print(f"{'slice':>10} {'share':>7} {'needed':>12} {'available':>14}")
    readers, problems = {}, []
    for name, (d, share, _is_code) in SLICES.items():
        p = Path(d)
        need = int(a.target_tokens * share)
        man = p / f"{a.split}-manifest.json"
        if not man.exists():
            problems.append(f"{name}: no {a.split}-manifest.json in {d} "
                            f"(directory exists: {p.exists()})")
            print(f"{name:>10} {share:>6.1%} {need:>12,} {'MISSING':>14}")
            continue
        r = SliceReader(name, p, a.split, eot)
        readers[name] = r
        flag = "" if r.total >= need else "  <-- SHORT"
        print(f"{name:>10} {share:>6.1%} {need:>12,} {r.total:>14,}{flag}")
        if r.total < need:
            problems.append(
                f"{name}: needs {need:,} tokens for a {share:.1%} share of "
                f"{a.target_tokens:,}, has {r.total:,} "
                f"({r.total / need:.2f}x). Lower --target-tokens, or rebuild the slice.")
    if problems:
        raise SystemExit("\nrefusing to build a mis-proportioned corpus:\n  " +
                         "\n  ".join(problems))

    names = list(SLICES)
    weights = np.array([SLICES[n][1] for n in names], dtype=np.float64)
    weights /= weights.sum()
    is_code = {n: SLICES[n][2] for n in names}

    state_path = out / f"mix-state-{a.split}.json"
    done = json.loads(state_path.read_text()) if state_path.exists() else None
    if done:
        for n, st in done["readers"].items():
            readers[n].restore(st)
        print(f"\nresuming at {done['written']:,} tokens, shard {done['shard']}")
    rng = np.random.default_rng(a.seed if not done else done["rng_seed"])
    written = done["written"] if done else 0
    shard_i = done["shard"] if done else 0
    val_written = done["val_written"] if done else 0
    per_slice = done["per_slice"] if done else dict.fromkeys(names, 0)
    per_slice_val = done.get("per_slice_val", dict.fromkeys(names, 0)) if done \
        else dict.fromkeys(names, 0)
    fim_applied = done["fim_applied"] if done else 0

    manifests = {"train": [], "val": []}
    for sp in ("train", "val"):
        p = out / f"{sp}-manifest.json"
        if p.exists() and done:
            manifests[sp] = json.loads(p.read_text())["shards"]

    buf: dict[str, list[int]] = {"train": [], "val": []}
    val_shard = 0

    def flush(sp: str, idx: int) -> int:
        if not buf[sp]:
            return idx
        name = f"{sp}-{idx:05d}.bin"
        np.asarray(buf[sp], dtype=np.uint16).tofile(out / name)
        manifests[sp].append({"file": name, "tokens": len(buf[sp])})
        (out / f"{sp}-manifest.json").write_text(json.dumps(
            {"split": sp, "dtype": "uint16", "shards": manifests[sp],
             "total_tokens": sum(s["tokens"] for s in manifests[sp])}, indent=1))
        buf[sp] = []
        return idx + 1

    exhausted: set[str] = set()
    while written < a.target_tokens:
        live = [i for i, n in enumerate(names) if n not in exhausted]
        if not live:
            raise SystemExit(f"every slice exhausted at {written:,}/{a.target_tokens:,}")
        # Pick by TOKEN DEFICIT, not by a weighted document draw. Document sizes differ by
        # more than an order of magnitude across slices — repo-packed code averages ~38k
        # tokens per document, web ~1k — so drawing documents in proportion to the target
        # share puts code at ~100% of the corpus instead of 66%. Measured, not theorised:
        # the first build of this script did exactly that. Choosing whichever slice is
        # furthest below its target share converges regardless of document size.
        total = max(written, 1)
        pick = names[max(live, key=lambda i: weights[i] * total - per_slice[names[i]])]
        doc = readers[pick].next_doc()
        if doc is None:
            exhausted.add(pick)
            print(f"  slice {pick!r} exhausted at {written:,} tokens", flush=True)
            continue
        ids = doc.tolist()
        if is_code[pick] and rng.random() < a.fim_rate:
            ids = fim_transform(ids, fim, rng)
            fim_applied += 1
        sp = "val" if val_written < a.val_tokens else "train"
        buf[sp].extend(ids)
        buf[sp].append(eot)
        n = len(ids) + 1
        if sp == "val":
            per_slice_val[pick] += n
            val_written += n
        else:
            per_slice[pick] += n
            written += n
        if len(buf[sp]) >= a.shard_tokens:
            if sp == "val":
                val_shard = flush("val", val_shard)
            else:
                shard_i = flush("train", shard_i)
                state_path.write_text(json.dumps({
                    "written": written, "val_written": val_written, "shard": shard_i,
                    "per_slice": per_slice, "per_slice_val": per_slice_val,
                    "fim_applied": fim_applied, "rng_seed": a.seed,
                    "readers": {n: r.state() for n, r in readers.items()}}))
                pct = written / a.target_tokens * 100
                comp = "  ".join(f"{n}:{per_slice[n]/max(written,1):.1%}" for n in names)
                print(f"  {written:,}/{a.target_tokens:,} ({pct:.1f}%) | {comp}", flush=True)

    flush("val", val_shard)
    shard_i = flush("train", shard_i)
    (out / "tokenizer.json").write_bytes(Path(a.tokenizer).read_bytes())
    composition = {n: {"tokens": per_slice[n], "share": per_slice[n] / max(written, 1),
                       "target_share": SLICES[n][1]} for n in names}
    (out / "composition.json").write_text(json.dumps(
        {"total_tokens": written, "val_tokens": val_written, "seed": a.seed,
         "fim_rate": a.fim_rate, "fim_documents": fim_applied,
         "tokenizer": Path(a.tokenizer).name, "slices": composition}, indent=1))
    state_path.write_text(json.dumps({
        "written": written, "val_written": val_written, "shard": shard_i,
        "per_slice": per_slice, "per_slice_val": per_slice_val,
        "fim_applied": fim_applied, "rng_seed": a.seed,
        "readers": {n: r.state() for n, r in readers.items()}}))
    print(f"\ndone: {written:,} train + {val_written:,} val tokens -> {out}")
    for n in names:
        print(f"  {n:>10} {per_slice[n]:>14,}  {per_slice[n]/written:6.2%} "
              f"(target {SLICES[n][1]:.1%})")
    print(f"  FIM applied to {fim_applied:,} documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
