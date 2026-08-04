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
import os
import shutil
import time
from pathlib import Path

import torch

from microlab.data.shard_dataset import ShardDataset
from microlab.tokenizer.fast import FastTokenizer
from microlab.train import distributed as dist_util
from microlab.train.trainer import CorruptCheckpoint, Trainer


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


def _remote_fetch(data_dir: str):
    """Callback that pulls one shard from S3-compatible storage into `data_dir`.

    Reads MICROLAB_SHARD_{BUCKET,PREFIX} and the B2_CORPUS_* credentials. One boto3 client
    is built once and closed over, because a client per shard would re-do TLS and
    credential resolution 105 times.
    """
    import boto3
    from botocore.config import Config

    bucket = os.environ["MICROLAB_SHARD_BUCKET"]
    prefix = os.environ.get("MICROLAB_SHARD_PREFIX", "mix-v1")
    s3 = boto3.client(
        "s3", endpoint_url=os.environ["B2_CORPUS_ENDPOINT"],
        aws_access_key_id=os.environ["B2_CORPUS_KEY_ID"],
        aws_secret_access_key=os.environ["B2_CORPUS_APPLICATION_KEY"],
        config=Config(retries={"max_attempts": 10, "mode": "adaptive"},
                      max_pool_connections=32))
    from boto3.s3.transfer import TransferConfig
    tcfg = TransferConfig(multipart_threshold=200 * 1024**2,
                          multipart_chunksize=200 * 1024**2, max_concurrency=16)

    def fetch(name: str, dest) -> None:
        t0 = time.time()
        s3.download_file(bucket, f"{prefix}/{name}", str(dest), Config=tcfg)
        mb = Path(dest).stat().st_size / 1e6
        print(f"  [shard] {name} {mb:.0f} MB in {time.time()-t0:.0f}s", flush=True)

    return fetch


def load_config(path: str):
    spec = importlib.util.spec_from_file_location("run_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="path to a config module (e.g. configs/150m.py)")
    ap.add_argument("--data-dir", default=None,
                    help="dir with train/val .bin shards; overrides cfg.data_dir when the "
                         "config does not set one. Passing a value that CONTRADICTS a "
                         "config's data_dir is an error, not a silent override.")
    ap.add_argument("--init-ckpt", default=None,
                    help="warm-start: load model WEIGHTS ONLY from this checkpoint "
                         "(fresh optimizer, step 0); out_dir must have no checkpoints")
    args = ap.parse_args()

    cfg = load_config(args.config)
    # Under torchrun this joins the process group and binds this rank to its GPU; a plain
    # `python scripts/pretrain.py` is unaffected and stays single-process.
    cfg.device = dist_util.setup(cfg.device)
    # Resolve the data dir from config and CLI. A contradiction is an ERROR: silently
    # letting the CLI win would mean a config could claim one corpus while the run used
    # another, which is exactly how the repetition lane's intervention became invisible.
    cfg_dd = getattr(cfg, "data_dir", None)
    if cfg_dd and args.data_dir and Path(cfg_dd) != Path(args.data_dir):
        raise SystemExit(
            f"data-dir conflict: config says {cfg_dd!r}, --data-dir says {args.data_dir!r}. "
            f"Pick one — a run must not disagree with its own config about what it trained on."
        )
    args.data_dir = cfg_dd or args.data_dir or "data/shards"
    # match the model's vocab to the tokenizer that produced the shards
    tok_path = Path(args.data_dir) / "tokenizer.json"
    tok = None
    if tok_path.exists():
        tok = FastTokenizer.load(str(tok_path))
        if cfg.vocab_size < tok.vocab_size:
            # The model could not emit the tokenizer's highest ids; raising to fit is the
            # safety property this override exists for.
            print(f"vocab_size raised to fit the tokenizer: "
                  f"{cfg.vocab_size} -> {tok.vocab_size}")
            cfg.vocab_size = tok.vocab_size
        elif cfg.vocab_size > tok.vocab_size:
            # A LARGER config vocab is a deliberate pad, not a mistake, and must not be
            # clobbered. Adding the 3 FIM sentinels took the tokenizer to 49,155, which is
            # divisible by neither 8 nor 64 — a badly shaped embedding/lm_head matmul on
            # every step of a multi-week run. The config pads to 49,280 = 385 x 128.
            # (This assignment previously overwrote the pad unconditionally.)
            print(f"vocab_size {cfg.vocab_size} > tokenizer {tok.vocab_size}: keeping the "
                  f"config's padding ({cfg.vocab_size - tok.vocab_size} unused rows)")
        else:
            print(f"vocab_size matches the tokenizer: {cfg.vocab_size}")

    # Make the run dir self-contained: co-locate the tokenizer with the checkpoints so the
    # console can serve this run without guessing which tokenizer produced it (decoding a
    # checkpoint with the wrong tokenizer yields garbage).
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if tok_path.exists() and not (out_dir / "tokenizer.json").exists():
        shutil.copy(tok_path, out_dir / "tokenizer.json")
        print(f"co-located tokenizer -> {out_dir / 'tokenizer.json'}")

    # Remote shards: fetch each one the first time training touches it, instead of
    # downloading the whole 39 GB corpus before step 1. On preemptible hardware that pull
    # is paid on EVERY re-provision — measured 9 min from US-West and 54 min from Asia —
    # and it is the only reason host choice was ever constrained by geography. Enabled by
    # setting MICROLAB_SHARD_BUCKET; absent, behaviour is unchanged.
    fetch = _remote_fetch(args.data_dir) if os.environ.get("MICROLAB_SHARD_BUCKET") else None
    train_ds = ShardDataset(args.data_dir, split="train", fetch=fetch)
    val_ds = ShardDataset(args.data_dir, split="val", fetch=fetch)
    if dist_util.is_main():
        print(f"train tokens={train_ds.total_tokens:,} val tokens={val_ds.total_tokens:,} "
              f"| world_size={dist_util.world_size()}")

    trainer = Trainer(cfg, train_ds, val_ds, tokenizer=tok)
    if args.init_ckpt is not None:
        warm_start(trainer, args.init_ckpt)  # raises if out_dir already has checkpoints
    else:
        ckpts = sorted(
            Path(cfg.out_dir).glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1])
        )
        # Walk DOWN on load failure instead of crashing on the newest. A truncated upload
        # can arrive as the highest-step checkpoint (the resume download only checks size
        # against the remote object, which IS the truncated size). Refusing to fall back
        # turns one bad file into a deterministic loop: re-provision, re-download the same
        # bytes, die again — at full GPU price per cycle. The bad file is renamed rather
        # than deleted so it stays diagnosable without ever being retried.
        loaded = None
        for c in reversed(ckpts):
            try:
                trainer.load_checkpoint(str(c))
            # ONLY unreadable files are skipped. A checkpoint that deserializes but does
            # not fit the model/optimizer is version skew between checkpoint and code,
            # and must surface as the error it is: this fallback once renamed a healthy
            # 9.2 GB checkpoint to .corrupt over an optimizer group-count change, then
            # reported "every local checkpoint failed" on a box that had nothing wrong
            # with its data. Skew errors propagate and fail the run with the real cause.
            except CorruptCheckpoint as e:
                # Rank 0 renames; the other ranks hit the same failure and just move on.
                # Every rank racing the same rename would crash the losers on
                # FileNotFoundError — the prune-unlink bug all over again.
                if dist_util.is_main():
                    bad = c.with_suffix(".pt.corrupt")
                    c.rename(bad)
                    print(f"CORRUPT checkpoint {c.name}: {e} — "
                          f"renamed to {bad.name}, trying the previous one", flush=True)
                continue
            print(f"resuming from {c} (latest loadable of {len(ckpts)} checkpoints)")
            loaded = c
            break
        if ckpts and loaded is None:
            raise SystemExit(
                "every local checkpoint failed to load — refusing to silently restart "
                "from step 0; delete the .corrupt files to do that explicitly")

    try:
        stats = trainer.train()
    finally:
        dist_util.teardown()
    if dist_util.is_main():
        ppl = math.exp(stats["val_loss"]) if stats.get("val_loss") is not None else float("nan")
        print(f"done: step={stats['step']} val_loss={stats['val_loss']:.3f} "
              f"perplexity={ppl:.2f}")


if __name__ == "__main__":
    main()
