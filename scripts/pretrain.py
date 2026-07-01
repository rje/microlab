"""Run a pretraining job from a config module, resumable across interruptions.

    python scripts/pretrain.py configs/150m.py --data-dir data/shards

If `<out_dir>/ckpt.pt` exists it resumes from there (checkpoints reproduce the
uninterrupted trajectory). Build the shards first with scripts/prepare_data.py.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from microlab.data.shard_dataset import ShardDataset
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
    train_ds = ShardDataset(args.data_dir, split="train")
    val_ds = ShardDataset(args.data_dir, split="val")
    print(f"train tokens={train_ds.total_tokens:,} val tokens={val_ds.total_tokens:,}")

    trainer = Trainer(cfg, train_ds, val_ds)
    ckpt = Path(cfg.out_dir) / "ckpt.pt"
    if ckpt.exists():
        print(f"resuming from {ckpt}")
        trainer.load_checkpoint(str(ckpt))

    stats = trainer.train()
    print("done:", {k: v for k, v in stats.items() if k != "history"})


if __name__ == "__main__":
    main()
