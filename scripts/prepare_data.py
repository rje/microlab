"""Tokenize a corpus ONCE into reusable uint16 .bin shards, so training never
re-tokenizes. Writes disjoint train/val shards + a manifest + the tokenizer, all under
`--out`; every later run (and every resume) memmaps these straight off disk.

Works with any streamed HuggingFace text dataset:

    # FineWeb-Edu (default) — the real pretraining corpus
    python scripts/prepare_data.py --out data/shards/fineweb --tokens 3_000_000_000

    # TinyStories — small, fluent, tractable
    python scripts/prepare_data.py --out data/shards/tinystories \
        --hf-dataset roneneldan/TinyStories --hf-config "" --tokens 100_000_000

Re-running with the same --out reuses the existing tokenizer (`<out>/tokenizer.json`).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from microlab.data.prepare import (
    batched_token_chunks,
    parallel_token_chunks,
    take_tokens,
    write_shards,
)
from microlab.tokenizer.fast import FastTokenizer


def stream_hf(dataset: str, config: str, split: str, text_field: str, max_docs: int | None = None):
    from datasets import load_dataset  # optional/heavy dep

    kwargs = {"split": split, "streaming": True}
    if config:
        kwargs["name"] = config
    ds = load_dataset(dataset, **kwargs)
    for i, row in enumerate(ds):
        if max_docs is not None and i >= max_docs:
            return
        text = row[text_field]
        if text and text.strip():
            yield text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/shards/fineweb")
    ap.add_argument("--hf-dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--hf-config", default="sample-10BT", help='"" for datasets without a config')
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--tokens", type=int, default=3_000_000_000, help="train token budget")
    ap.add_argument("--val-tokens", type=int, default=5_000_000)
    ap.add_argument("--shard-size", type=int, default=100_000_000)
    ap.add_argument("--batch-docs", type=int, default=256,
                    help="documents per encode_batch call")
    ap.add_argument("--workers", type=int, default=max(1, min(24, (os.cpu_count() or 4) - 2)),
                    help="tokenizer worker processes (1 = single-process). ~3x faster than one "
                         "process since encode_batch's rayon parallelism doesn't scale")
    ap.add_argument("--tokenizer-sample-docs", type=int, default=50_000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok_path = out / "tokenizer.json"

    if tok_path.exists():
        tok = FastTokenizer.load(str(tok_path))
        print(f"reusing tokenizer {tok_path} (vocab {tok.vocab_size})")
    else:
        print(f"training {args.vocab_size}-vocab tokenizer on ~{args.tokenizer_sample_docs} docs")
        sample = list(stream_hf(args.hf_dataset, args.hf_config, args.split, args.text_field,
                                max_docs=args.tokenizer_sample_docs))
        tok = FastTokenizer.train(sample, vocab_size=args.vocab_size, save_path=str(tok_path))
        print(f"saved tokenizer -> {tok_path} (vocab {tok.vocab_size})")

    eot = tok.eot_token
    docs = stream_hf(args.hf_dataset, args.hf_config, args.split, args.text_field)
    if args.workers > 1:
        chunks = parallel_token_chunks(tok_path, docs, eot, workers=args.workers,
                                       batch_docs=args.batch_docs)
    else:
        chunks = batched_token_chunks(tok, docs, eot, batch_docs=args.batch_docs)
    carry: list = []  # boundary remainder shared between the val and train segments
    print(f"writing {args.val_tokens:,} val tokens ({args.workers} workers x {args.batch_docs} "
          f"docs)...")
    write_shards(take_tokens(chunks, args.val_tokens, carry), str(out), split="val",
                 shard_size=args.shard_size)
    print(f"writing {args.tokens:,} train tokens (same stream continues; disjoint from val)")
    write_shards(take_tokens(chunks, args.tokens, carry), str(out), split="train",
                 shard_size=args.shard_size)
    print(f"done. reusable shards + tokenizer under {out}/")


if __name__ == "__main__":
    main()
