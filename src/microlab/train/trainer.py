"""Production Trainer for real-scale pretraining (Phase-2): bf16 autocast on CUDA, AdamW
with a linear-warmup + cosine-decay LR schedule, gradient accumulation and clipping, and
checkpoint save/resume that reproduces the uninterrupted training trajectory (model,
optimizer, step counter, and BOTH the global torch RNG and the data-sampling generator)."""

from __future__ import annotations

import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from microlab.data.reference.dataset import get_batch
from microlab.model.reference.sample import generate
from microlab.model.reference.train import _resolve_device
from microlab.model.reference.variants import VariantConfig, VariantGPT
from microlab.train.config import RunConfig

# Fixed prompt used for sample-text logging so generations are comparable across steps.
SAMPLE_PROMPT = "\n"
SAMPLE_TOKENS = 64


def get_lr(step: int, cfg: RunConfig) -> float:
    """nanoGPT LR schedule: linear warmup 0 -> cfg.lr over `warmup_steps`, then cosine
    decay cfg.lr -> cfg.min_lr over the remaining `lr_decay_steps - warmup_steps`, then
    flat at cfg.min_lr."""
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    if step >= cfg.lr_decay_steps:
        return cfg.min_lr
    decay_ratio = (step - cfg.warmup_steps) / (cfg.lr_decay_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


class TensorData:
    """Self-contained data adapter wrapping a 1-D LongTensor. Exposes the same
    `get_batch` contract as the real ShardDataset so tests need no data files."""

    def __init__(self, tokens: torch.Tensor) -> None:
        self.tokens = tokens

    def get_batch(
        self,
        block_size: int,
        batch_size: int,
        device: str,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return get_batch(self.tokens, block_size, batch_size, device, generator)


class Trainer:
    """Drives a real pretraining run from a single RunConfig.

    The `train_data` / `val_data` objects only need to expose
    `get_batch(block_size, batch_size, device, generator) -> (x, y)`.
    """

    def __init__(self, cfg: RunConfig, train_data, val_data=None, tokenizer=None) -> None:
        self.cfg = cfg
        self.train_data = train_data
        self.val_data = val_data
        # Used only for sample-text logging (add_text of generated completions); never
        # touches the training math.
        self.tokenizer = tokenizer
        self.device = _resolve_device(cfg.device)
        # Seed BEFORE building the model so weight init is deterministic and identical
        # across runs regardless of ambient global RNG state (the resume test relies on
        # runs a/b/c starting from bit-identical parameters).
        torch.manual_seed(cfg.seed)
        self.model = VariantGPT(
            VariantConfig(
                vocab_size=cfg.vocab_size,
                block_size=cfg.block_size,
                n_layer=cfg.n_layer,
                n_head=cfg.n_head,
                n_embd=cfg.n_embd,
                dropout=cfg.dropout,
                norm=cfg.norm,
                pos=cfg.pos,
                mlp=cfg.mlp,
            )
        )
        self.model.to(self.device)
        self.raw_model = self.model  # state_dict source of truth (survives torch.compile)
        self.raw_model.grad_checkpoint = cfg.grad_checkpoint
        if cfg.compile:
            self.model = torch.compile(self.model)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )
        # Separate CPU generator drives batch sampling, so data order is reproducible and
        # independent of any global-RNG consumption during forward/backward (e.g. dropout).
        self.data_gen = torch.Generator().manual_seed(cfg.seed)
        self.use_amp = self.device.startswith("cuda") and cfg.dtype == "bfloat16"
        self.step = 0
        # Last-step telemetry captured by train_step for side-effect-only TB logging.
        self.last_lr = cfg.lr
        self.last_grad_norm = 0.0

    def _autocast(self):
        if self.use_amp:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def train_step(self) -> float:
        cfg = self.cfg
        lr = get_lr(self.step, cfg)
        self.last_lr = lr
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(cfg.grad_accum):
            x, y = self.train_data.get_batch(
                cfg.block_size, cfg.batch_size, self.device, self.data_gen
            )
            with self._autocast():
                _, loss = self.model(x, y)
                loss = loss / cfg.grad_accum
            loss.backward()
            total_loss += loss.item()
        if cfg.grad_clip > 0:
            # clip_grad_norm_ returns the total grad L2 norm computed before clipping;
            # capturing it is side-effect free (the clip itself is unchanged).
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
        else:
            # No clipping: read the norm without mutating grads (max_norm=inf scales by 1.0).
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float("inf"))
        self.last_grad_norm = float(grad_norm)
        self.optimizer.step()
        self.step += 1
        return total_loss

    @torch.no_grad()
    def estimate_val(self) -> float | None:
        if self.val_data is None:
            return None
        cfg = self.cfg
        was_training = self.model.training
        self.model.eval()
        # Fresh, deterministic generator: val loss is stable across steps and does not
        # perturb the training data generator.
        gen = torch.Generator().manual_seed(cfg.seed)
        total = 0.0
        for _ in range(cfg.eval_iters):
            x, y = self.val_data.get_batch(cfg.block_size, cfg.batch_size, self.device, gen)
            with self._autocast():
                _, loss = self.model(x, y)
            total += loss.item()
        if was_training:
            self.model.train()
        return total / cfg.eval_iters

    def save_checkpoint(self, path: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        ckpt = {
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "torch_rng_state": torch.get_rng_state(),
            "data_gen_state": self.data_gen.get_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "cfg": self.cfg,
        }
        torch.save(ckpt, path)

    def _prune_checkpoints(self) -> None:
        keep = self.cfg.ckpt_keep
        if keep <= 0:
            return
        ckpts = sorted(
            Path(self.cfg.out_dir).glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1])
        )
        for stale in ckpts[:-keep]:
            stale.unlink()

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.raw_model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step = ckpt["step"]
        # RNG state tensors must live on CPU for set_rng_state / generator.set_state.
        torch.set_rng_state(ckpt["torch_rng_state"].cpu())
        self.data_gen.set_state(ckpt["data_gen_state"].cpu())
        if ckpt.get("cuda_rng_state") is not None and torch.cuda.is_available():
            # map_location moved these ByteTensors onto CUDA; RNG state must be CPU bytes.
            torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["cuda_rng_state"]])

    @torch.no_grad()
    def _log_sample(self, writer, step: int) -> None:
        """Greedy-decode a short completion from a fixed prompt for add_text. Greedy
        (temperature=0) consumes no RNG, so this observational logging never perturbs the
        training trajectory. Restores train() mode that generate() flips to eval()."""
        ids = self.tokenizer.encode(SAMPLE_PROMPT) or [0]
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        was_training = self.raw_model.training
        out = generate(self.raw_model, idx, SAMPLE_TOKENS, temperature=0.0)
        if was_training:
            self.raw_model.train()
        writer.add_text("samples", self.tokenizer.decode(out[0].tolist()), step)

    def train(self) -> dict:
        cfg = self.cfg
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        # Guarded so training is a no-op-logging run when tensorboard isn't importable.
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=cfg.out_dir)
        except Exception:
            writer = None
        tokens_per_step = cfg.batch_size * cfg.grad_accum * cfg.block_size
        last_log_time = time.perf_counter()
        last_log_step = self.step
        self.model.train()
        history: list[float] = []
        val_loss: float | None = None
        while self.step < cfg.max_steps:
            loss = self.train_step()
            history.append(loss)
            step = self.step  # already incremented by train_step
            if cfg.eval_interval > 0 and step % cfg.eval_interval == 0:
                val_loss = self.estimate_val()
                if writer is not None and val_loss is not None:
                    writer.add_scalar("val/loss", val_loss, step)
                    writer.add_scalar("val/perplexity", math.exp(val_loss), step)
                    if self.tokenizer is not None:
                        self._log_sample(writer, step)
            if cfg.log_interval > 0 and step % cfg.log_interval == 0:
                print(f"step {step}/{cfg.max_steps} loss {loss:.4f} lr {get_lr(step, cfg):.2e}")
                if writer is not None:
                    now = time.perf_counter()
                    dt = now - last_log_time
                    tps = tokens_per_step * (step - last_log_step) / dt if dt > 0 else 0.0
                    writer.add_scalar("train/loss", loss, step)
                    writer.add_scalar("lr", self.last_lr, step)
                    writer.add_scalar("train/tokens_per_sec", tps, step)
                    writer.add_scalar("train/grad_norm", self.last_grad_norm, step)
                    last_log_time, last_log_step = now, step
            if cfg.ckpt_interval > 0 and step % cfg.ckpt_interval == 0:
                self.save_checkpoint(os.path.join(cfg.out_dir, f"ckpt_{step}.pt"))
                self._prune_checkpoints()
        if self.val_data is not None and val_loss is None:
            val_loss = self.estimate_val()
        if writer is not None:
            writer.close()
        return {
            "final_loss": history[-1] if history else None,
            "history": history,
            "val_loss": val_loss,
            "step": self.step,
        }
