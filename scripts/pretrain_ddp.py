"""Multi-GPU data-parallel pretraining via torchrun. Reuses the single-GPU Trainer;
each rank samples its own data stream (seed+rank), gradients sync through DDP, rank 0
logs/checkpoints. Scaling-efficiency drill for the Phase 7 cloud rung.

    torchrun --nproc_per_node=4 scripts/pretrain_ddp.py configs/150m.py \
        --data-dir data/shards/tinystories
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402

from microlab.data.shard_dataset import ShardDataset  # noqa: E402
from microlab.tokenizer.fast import FastTokenizer  # noqa: E402
from microlab.train.trainer import Trainer, get_lr  # noqa: E402


def load_config(path: str):
    spec = importlib.util.spec_from_file_location("run_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config


class DDPTrainer(Trainer):
    """Trainer whose train_step syncs grads across ranks (no_sync on all but the last
    micro-step) and averages the logged loss over the world."""

    def __init__(self, cfg, train_ds, val_ds, tokenizer, rank: int) -> None:
        super().__init__(cfg, train_ds, val_ds, tokenizer=tokenizer)
        self.rank = rank
        self.data_gen = torch.Generator().manual_seed(cfg.seed + rank)  # shard by stream
        self.ddp = DDP(self.model, device_ids=[rank])

    def load_checkpoint(self, path: str) -> None:
        super().load_checkpoint(path)
        # The checkpoint restores rank-0's data generator onto every rank, which would make
        # all ranks draw identical batches after a resume. Re-seed per rank, mixing in the
        # resumed step so the stream also differs from the pre-crash stream.
        self.data_gen = torch.Generator().manual_seed(self.cfg.seed + self.rank + self.step)

    def train_step(self) -> float:
        cfg = self.cfg
        lr = get_lr(self.step, cfg)
        self.last_lr = lr
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.ddp.train()
        self.optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for micro in range(cfg.grad_accum):
            x, y = self.train_data.get_batch(cfg.block_size, cfg.batch_size,
                                             self.device, self.data_gen)
            sync = micro == cfg.grad_accum - 1
            ctx = self.ddp.no_sync() if not sync else torch.enable_grad()
            with ctx, self._autocast():
                _, loss = self.ddp(x, y)
                loss = loss / cfg.grad_accum
            loss.backward()
            total += loss.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.ddp.parameters(), cfg.grad_clip if cfg.grad_clip > 0 else float("inf"))
        self.last_grad_norm = float(grad_norm)
        self.optimizer.step()
        self.step += 1
        t = torch.tensor(total, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        return t.item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--data-dir", default="data/shards")
    args = ap.parse_args()

    # Single-node assumption: LOCAL_RANK doubles as the global rank for data sharding
    # (multi-node would need dist.get_rank() to keep per-rank data streams distinct).
    rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)

    cfg = load_config(args.config)
    cfg.device = f"cuda:{rank}"
    tok_path = Path(args.data_dir) / "tokenizer.json"
    tok = FastTokenizer.load(str(tok_path)) if tok_path.exists() else None
    if tok is not None:
        cfg.vocab_size = tok.vocab_size
    if rank != 0:
        cfg.log_interval = 0
        cfg.ckpt_interval = 0
        cfg.eval_interval = 0

    train_ds = ShardDataset(args.data_dir, split="train")
    val_ds = ShardDataset(args.data_dir, split="val")
    trainer = DDPTrainer(cfg, train_ds, val_ds, tok, rank)
    ckpts = sorted(Path(cfg.out_dir).glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if ckpts:
        trainer.load_checkpoint(str(ckpts[-1]))
        if rank == 0:
            print(f"resumed from {ckpts[-1]}")
    stats = trainer.train()
    if rank == 0:
        ppl = math.exp(stats["val_loss"]) if stats.get("val_loss") is not None else float("nan")
        print(f"done: step={stats['step']} val_loss={stats['val_loss']} ppl={ppl:.2f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
