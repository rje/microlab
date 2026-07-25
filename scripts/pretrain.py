"""Run a pretraining job from a config module, resumable across interruptions.

    python scripts/pretrain.py configs/150m.py --data-dir data/shards

If `<out_dir>/ckpt.pt` exists it resumes from there (checkpoints reproduce the
uninterrupted trajectory). Build the shards first with scripts/prepare_data.py.

`--init-ckpt PATH` warm-starts instead: MODEL WEIGHTS ONLY are loaded from PATH (e.g. a
scripts/convert_gqa.py output) into the run-config model — fresh optimizer, step 0, no
RNG restore. Mutually exclusive with an out_dir that already has checkpoints (resume and
warm-start disagree about optimizer/step/RNG state; we raise rather than guess).
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import shutil
from pathlib import Path

import torch

from microlab.data.shard_dataset import ShardDataset
from microlab.tokenizer.fast import FastTokenizer
from microlab.train.trainer import Trainer


def warm_start(trainer: Trainer, init_ckpt: str) -> None:
    """Load MODEL WEIGHTS ONLY from `init_ckpt` into the trainer's model.

    The optimizer stays fresh (a converted checkpoint has no/incompatible optimizer
    state) and the step counter stays 0 — this is the start of a NEW run, not a resume.
    Authority rule: the RUN CONFIG wins. The trainer's model is already built from the
    run config; the checkpoint's embedded cfg is metadata only. The strict
    load_state_dict below is the shape-compatibility assertion — a checkpoint whose
    weights don't fit the run-config model fails loudly here.
    """
    existing = sorted(Path(trainer.cfg.out_dir).glob("ckpt_*.pt"))
    if existing:
        raise RuntimeError(
            f"--init-ckpt given but out_dir {trainer.cfg.out_dir} already has checkpoints "
            f"({existing[0].name}...): refusing to guess between warm-start and resume. "
            "Use a fresh out_dir, or drop --init-ckpt to resume."
        )
    ckpt = torch.load(init_ckpt, map_location="cpu", weights_only=False)
    trainer.raw_model.load_state_dict(ckpt["model"])
    assert trainer.step == 0
    print(f"warm start: model weights <- {init_ckpt} (fresh optimizer, step 0)")


def load_config(path: str):
    spec = importlib.util.spec_from_file_location("run_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="path to a config module (e.g. configs/150m.py)")
    ap.add_argument("--data-dir", default="data/shards", help="dir with train/val .bin shards")
    ap.add_argument("--init-ckpt", default=None,
                    help="warm-start: load model WEIGHTS ONLY from this checkpoint "
                         "(fresh optimizer, step 0); out_dir must have no checkpoints")
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
    if args.init_ckpt is not None:
        warm_start(trainer, args.init_ckpt)  # raises if out_dir already has checkpoints
    else:
        ckpts = sorted(
            Path(cfg.out_dir).glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1])
        )
        if ckpts:
            print(f"resuming from {ckpts[-1]} (latest of {len(ckpts)} checkpoints)")
            trainer.load_checkpoint(str(ckpts[-1]))

    stats = trainer.train()
    ppl = math.exp(stats["val_loss"]) if stats.get("val_loss") is not None else float("nan")
    print(f"done: step={stats['step']} val_loss={stats['val_loss']:.3f} perplexity={ppl:.2f}")


if __name__ == "__main__":
    main()
