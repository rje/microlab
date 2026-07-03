"""Run a pretraining job from a config module, resumable across interruptions.

    python scripts/pretrain.py configs/150m.py --data-dir data/shards

If `<out_dir>/ckpt.pt` exists it resumes from there (checkpoints reproduce the
uninterrupted trajectory). Build the shards first with scripts/prepare_data.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import shutil
from pathlib import Path

from microlab.data.shard_dataset import ShardDataset
from microlab.tokenizer.fast import FastTokenizer
from microlab.train.trainer import Trainer


def load_config(path: str):
    spec = importlib.util.spec_from_file_location("run_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="path to a config module (e.g. configs/150m.py)")
    ap.add_argument("--data-dir", default="data/shards", help="dir with train/val .bin shards")
    args = ap.parse_args()

    cfg = load_config(args.config)
    # match the model's vocab to the tokenizer that produced the shards
    tok_path = Path(args.data_dir) / "tokenizer.json"
    tok = None
    if tok_path.exists():
        tok = FastTokenizer.load(str(tok_path))
        cfg.vocab_size = tok.vocab_size
        print(f"vocab_size set from tokenizer: {cfg.vocab_size}")

    # Make the run dir self-contained: co-locate the tokenizer with the checkpoints so the
    # console can serve this run without guessing which tokenizer produced it (decoding a
    # checkpoint with the wrong tokenizer yields garbage).
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if tok_path.exists() and not (out_dir / "tokenizer.json").exists():
        shutil.copy(tok_path, out_dir / "tokenizer.json")
        print(f"co-located tokenizer -> {out_dir / 'tokenizer.json'}")

    train_ds = ShardDataset(args.data_dir, split="train")
    val_ds = ShardDataset(args.data_dir, split="val")
    print(f"train tokens={train_ds.total_tokens:,} val tokens={val_ds.total_tokens:,}")

    trainer = Trainer(cfg, train_ds, val_ds, tokenizer=tok)
    ckpts = sorted(Path(cfg.out_dir).glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if ckpts:
        print(f"resuming from {ckpts[-1]} (latest of {len(ckpts)} checkpoints)")
        trainer.load_checkpoint(str(ckpts[-1]))

    stats = trainer.train()
    ppl = math.exp(stats["val_loss"]) if stats.get("val_loss") is not None else float("nan")
    print(f"done: step={stats['step']} val_loss={stats['val_loss']:.3f} perplexity={ppl:.2f}")


if __name__ == "__main__":
    main()
