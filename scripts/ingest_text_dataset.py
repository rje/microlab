#!/usr/bin/env python
"""Stream a HuggingFace text dataset into code-49k shards.

The corpus builder handles the-stack (per-language parquet, license gates, attribution);
the retokeniser handles corpora we already tokenised. This handles the third case: a plain
text dataset on the Hub that we want in our vocabulary and shard format.

Used for the non-code slices of the mix — open-web-math, common-pile/arxiv_papers,
bigcode/commitpackft — each of which arrives as text and needs the same treatment.

Streams (never materialises the dataset), writes shards progressively, resumable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

COMMIT_FIELDS = ("message", "subject", "old_file", "new_file",
                 "old_contents", "new_contents")


def render_commit(rec: dict) -> str:
    """Render a CommitPack record as one pretraining document.

    Copying a single text column is wrong for this corpus. `new_contents` alone is just
    more source code, which the code slice already supplies 65% of; the thing a commits
    slice uniquely teaches is the PAIRING of a natural-language intent with the edit that
    realises it. So the document keeps the message and both sides of the change.

    Missing fields raise rather than degrade to a partial document: a silently
    message-less "commit" would train the model on exactly the association we are paying
    for, minus the supervision.
    """
    missing = [f for f in COMMIT_FIELDS if not isinstance(rec.get(f), str)]
    if missing:
        raise KeyError(f"commit record lacks {missing}; got columns {sorted(rec)}")
    # `message` is the full text and usually starts with `subject`; prefer it, and only
    # add `subject` when it genuinely carries something message does not.
    msg = rec["message"].strip() or rec["subject"].strip()
    return (f"{msg}\n\n"
            f"--- a/{rec['old_file']}\n{rec['old_contents']}\n"
            f"+++ b/{rec['new_file']}\n{rec['new_contents']}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="HF dataset id")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--text-field", default=None,
                    help="column holding the text; auto-detected when omitted")
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-files", default=None,
                    help="load these files directly (a URL/glob) instead of resolving the "
                         "dataset by id. Needed for repos that ship a loading SCRIPT, which "
                         "`datasets` refuses to execute: bigcode/commitpackft fails with "
                         "'Dataset scripts are no longer supported, but found "
                         "commitpackft.py' even though its data is plain JSONL.")
    ap.add_argument("--format", default="auto", choices=("auto", "commit"),
                    help="'commit' renders a CommitPack-style record (subject + before/after "
                         "contents) instead of copying one text column")
    ap.add_argument("--hf-split", default="train")
    ap.add_argument("--max-tokens", type=int, default=0, help="0 = whole dataset")
    ap.add_argument("--val-tokens", type=int, default=10_000_000)
    ap.add_argument("--shard-tokens", type=int, default=100_000_000)
    ap.add_argument("--batch-docs", type=int, default=1000)
    a = ap.parse_args()

    from datasets import load_dataset

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(a.tokenizer)
    eot = tok.token_to_id("<|endoftext|>")
    if eot is None:
        raise SystemExit("tokenizer has no <|endoftext|>")

    state_path = out / "ingest-state.json"
    done = json.loads(state_path.read_text()) if state_path.exists() else \
        {"rows": 0, "train_shard": 0, "train_tokens": 0, "val_tokens": 0}
    if done["train_tokens"] and a.max_tokens and done["train_tokens"] >= a.max_tokens:
        print(f"already complete: {done['train_tokens']:,} tokens")
        return 0

    if a.data_files:
        ds = load_dataset("json", data_files=a.data_files, split=a.hf_split, streaming=True)
    else:
        ds = load_dataset(a.dataset, a.config, split=a.hf_split, streaming=True)
    first = next(iter(ds))

    if a.format == "commit":
        render = render_commit
        render(first)                       # fail now, not 2 hours into the stream
        print(f"format: commit  (columns: {list(first)})", flush=True)
    else:
        field = a.text_field or next(
            (k for k in ("text", "content", "markdown", "new_contents", "message")
             if k in first and isinstance(first[k], str)), None)
        if field is None:
            raise SystemExit(
                f"could not find a text column in {list(first)}; pass --text-field")
        print(f"text column: {field!r}  (columns: {list(first)})", flush=True)

        def render(rec, _f=field):
            return rec.get(_f)

    manifests = {"train": [], "val": []}
    for sp in ("train", "val"):
        p = out / f"{sp}-manifest.json"
        if p.exists():
            manifests[sp] = json.loads(p.read_text())["shards"]

    buf: dict[str, list[int]] = {"train": [], "val": []}
    shard_i = {"train": done["train_shard"], "val": 0}
    written = {"train": done["train_tokens"], "val": done["val_tokens"]}

    def flush(sp: str):
        if not buf[sp]:
            return
        name = f"{sp}-{shard_i[sp]:05d}.bin"
        np.asarray(buf[sp], dtype=np.uint16).tofile(out / name)
        manifests[sp].append({"file": name, "tokens": len(buf[sp])})
        (out / f"{sp}-manifest.json").write_text(json.dumps(
            {"split": sp, "dtype": "uint16", "shards": manifests[sp],
             "total_tokens": sum(s["tokens"] for s in manifests[sp])}, indent=1))
        shard_i[sp] += 1
        buf[sp] = []

    texts, rows = [], 0
    target = a.max_tokens or float("inf")
    for rec in ds:
        rows += 1
        if rows <= done["rows"]:
            continue
        t = render(rec)
        if not isinstance(t, str) or not t.strip():
            continue
        texts.append(t)
        if len(texts) < a.batch_docs:
            continue
        for enc in tok.encode_batch(texts):
            # val first, then everything else to train — val is a held-out prefix, which is
            # fine for a supplementary slice (the code corpus routes by content hash).
            sp = "val" if written["val"] < a.val_tokens else "train"
            buf[sp].extend(enc.ids)
            buf[sp].append(eot)
            written[sp] += len(enc.ids) + 1
            if len(buf[sp]) >= a.shard_tokens:
                flush(sp)
        texts = []
        if written["train"] >= target:
            break
        if shard_i["train"] and not written["train"] % a.shard_tokens:
            pass
        state_path.write_text(json.dumps(
            {"rows": rows, "train_shard": shard_i["train"],
             "train_tokens": written["train"], "val_tokens": written["val"]}))
        if rows % 50_000 == 0:
            print(f"  {rows:,} rows | train {written['train']:,} "
                  f"| val {written['val']:,}", flush=True)
    if texts:
        for enc in tok.encode_batch(texts):
            sp = "val" if written["val"] < a.val_tokens else "train"
            buf[sp].extend(enc.ids)
            buf[sp].append(eot)
            written[sp] += len(enc.ids) + 1
    flush("train")
    flush("val")
    (out / "tokenizer.json").write_bytes(Path(a.tokenizer).read_bytes())
    state_path.write_text(json.dumps(
        {"rows": rows, "train_shard": shard_i["train"],
         "train_tokens": written["train"], "val_tokens": written["val"]}))
    print(f"done: train {written['train']:,} | val {written['val']:,} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    # os._exit rather than a normal return: HF `datasets` streaming leaves aiohttp/pyarrow
    # worker threads alive, and CPython's finalizer aborts with
    #   Fatal Python error: PyGILState_Release: thread state must be current when releasing
    # AFTER all data is safely written. The first run of this script wrote 3.0B math tokens,
    # printed "done:", then core-dumped on shutdown — which under `set -e` killed the arxiv
    # and commits ingests that were queued behind it. Skipping finalisation avoids a crash
    # that can only happen once the work is complete.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
