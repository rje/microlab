"""Tokenize a corpus into disjoint train/val uint16 .bin shards for pretraining.

Default source: FineWeb-Edu (`sample-10BT`, streamed — no full download). Trains a 32k
byte-level BPE on a sample if no tokenizer is given, then writes a held-out val split
followed by the train split from the SAME stream (so they don't overlap).

    python scripts/prepare_data.py --out data/shards --tokens 3_000_000_000
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from microlab.data.prepare import write_shards
from microlab.tokenizer.fast import FastTokenizer


def stream_fineweb(max_docs: int | None = None):
    from datasets import load_dataset  # optional/heavy dep

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
    )
    for i, row in enumerate(ds):
        if max_docs is not None and i >= max_docs:
            return
        text = row["text"]
        if text and text.strip():
            yield text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/shards")
    ap.add_argument("--tokenizer", default=None, help="tokenizer.json path (trained if absent)")
    ap.add_argument("--tokens", type=int, default=3_000_000_000, help="train token budget")
    ap.add_argument("--val-tokens", type=int, default=5_000_000)
    ap.add_argument("--shard-size", type=int, default=100_000_000)
    ap.add_argument("--tokenizer-sample-docs", type=int, default=50_000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.tokenizer and Path(args.tokenizer).exists():
        tok = FastTokenizer.load(args.tokenizer)
        print(f"loaded tokenizer {args.tokenizer} (vocab {tok.vocab_size})")
    else:
        print(f"training 32k tokenizer on ~{args.tokenizer_sample_docs} docs...")
        sample = list(stream_fineweb(max_docs=args.tokenizer_sample_docs))
        tok = FastTokenizer.train(sample, vocab_size=32000, save_path=str(out / "tokenizer.json"))
        print(f"trained tokenizer -> {out / 'tokenizer.json'} (vocab {tok.vocab_size})")

    eot = tok.eot_token

    def all_tokens():
        for text in stream_fineweb():
            yield from tok.encode(text)
            yield eot

    gen = all_tokens()
    print("writing val shards...")
    write_shards(itertools.islice(gen, args.val_tokens), str(out), split="val",
                 shard_size=args.shard_size)
    print("writing train shards (continuing the same stream, so val/train are disjoint)...")
    write_shards(itertools.islice(gen, args.tokens), str(out), split="train",
                 shard_size=args.shard_size)
    print("done.")


if __name__ == "__main__":
    main()
