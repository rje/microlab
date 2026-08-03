"""Production Trainer for real-scale pretraining (Phase-2): bf16 autocast on CUDA, AdamW
with a linear-warmup + cosine-decay LR schedule, gradient accumulation and clipping, and
checkpoint save/resume that reproduces the uninterrupted training trajectory (model,
optimizer, step counter, and BOTH the global torch RNG and the data-sampling generator)."""

from __future__ import annotations

import contextlib
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
from microlab.train import distributed as dist_util
from microlab.train.config import RunConfig
from microlab.train.muon import build_muon_optimizer

# Fixed prompt used for sample-text logging so generations are comparable across steps.
SAMPLE_PROMPT = "\n"
SAMPLE_TOKENS = 64


def gpu_scalars(device: str, include_nvml: bool) -> dict[str, float]:
    """TensorBoard GPU-telemetry scalars for `device`; empty off-CUDA (so CPU runs and the CPU
    test suite never touch torch.cuda.*). Memory metrics need only torch. The NVML metrics
    (temperature/utilization/power/clock) are added when `include_nvml` — i.e. nvidia-ml-py is
    importable and the init probe succeeded. Units: memory GB, temp C, util %, power W, clk MHz."""
    if not device.startswith("cuda"):
        return {}
    s = {
        "gpu/mem_allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "gpu/mem_reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "gpu/mem_max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    if include_nvml:
        s["gpu/temperature_c"] = float(torch.cuda.temperature())
        s["gpu/utilization_pct"] = float(torch.cuda.utilization())
        s["gpu/power_w"] = torch.cuda.power_draw() / 1000.0  # NVML reports milliwatts
        s["gpu/sm_clock_mhz"] = float(torch.cuda.clock_rate())
    return s


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
        # A milestone cadence that isn't a multiple of ckpt_interval would mean milestone
        # steps are never actually checkpointed (no ckpt_*.pt exists at those steps), so the
        # permanent trajectory would silently be empty. Fail loudly rather than mask it.
        if (
            cfg.ckpt_milestone_interval > 0
            and cfg.ckpt_interval > 0
            and cfg.ckpt_milestone_interval % cfg.ckpt_interval != 0
        ):
            raise ValueError(
                f"ckpt_milestone_interval ({cfg.ckpt_milestone_interval}) must be a multiple "
                f"of ckpt_interval ({cfg.ckpt_interval}); otherwise milestone steps are never "
                f"checkpointed and no permanent trajectory is saved."
            )
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
                # getattr: cfg may be unpickled from a checkpoint written before these
                # fields existed; the defaults reproduce that era's behavior exactly.
                n_kv_head=getattr(cfg, "n_kv_head", None),
                rope_base=getattr(cfg, "rope_base", 10000.0),
                block_norm=getattr(cfg, "block_norm", "pre"),
                hybrid_every=getattr(cfg, "hybrid_every", None),
                gdn_chunk=getattr(cfg, "gdn_chunk", 64),
                gdn_conv_kernel=getattr(cfg, "gdn_conv_kernel", 4),
                # Every architecture field must be threaded here or the config silently
                # cannot reach the model. This is the THIRD time that gap has bitten
                # (docs/sota-parity-1b.md #8: the 1B shipped MHA because RunConfig never
                # exposed n_kv_head). Keep in sync with VariantConfig.
                gdn_fused=getattr(cfg, "gdn_fused", True),
                gdn_gate=getattr(cfg, "gdn_gate", "scalar"),
                global_attn=getattr(cfg, "global_attn", "gqa"),
                mla_kv_lora=getattr(cfg, "mla_kv_lora", 512),
                qk_norm=getattr(cfg, "qk_norm", False),
            )
        )
        self.model.to(self.device)
        self.raw_model = self.model  # state_dict source of truth (survives torch.compile)
        self.raw_model.grad_checkpoint = cfg.grad_checkpoint
        self.raw_model.fused_ce = getattr(cfg, "fused_ce", False)
        # TF32 for any fp32 matmuls autocast leaves alone (the head, some reductions) — free.
        if self.device.startswith("cuda"):
            torch.set_float32_matmul_precision("high")
        if cfg.compile:
            self.model = torch.compile(self.model, mode=cfg.compile_mode)
        # DDP wraps AFTER compile, and `raw_model` stays the UNWRAPPED module — it is the
        # state_dict source of truth, so a checkpoint written at one world size loads at
        # any other. That property is what makes a run migratable mid-flight.
        self.ddp = None
        if dist_util.is_distributed():
            from torch.nn.parallel import DistributedDataParallel
            dev_ids = [dist_util.local_rank()] if self.device.startswith("cuda") else None
            self.ddp = DistributedDataParallel(self.model, device_ids=dev_ids)
            self.model = self.ddp
        if cfg.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=cfg.lr,
                betas=cfg.betas,
                weight_decay=cfg.weight_decay,
                fused=self.device.startswith("cuda"),  # fused CUDA optimizer step
            )
            # Uniform schedule scale; MuonAdamW stamps its own per-group lr_scale.
            for group in self.optimizer.param_groups:
                group["lr_scale"] = 1.0
        elif cfg.optimizer == "muon":
            self.optimizer = build_muon_optimizer(
                self.raw_model,  # named grouping needs the raw module, not the compile wrapper
                lr=cfg.lr,
                muon_lr=cfg.muon_lr,
                betas=cfg.betas,
                weight_decay=cfg.weight_decay,
                fused=self.device.startswith("cuda"),
            )
        else:
            raise ValueError(f"unknown optimizer {cfg.optimizer!r}; expected 'adamw' or 'muon'")
        # Separate CPU generator drives batch sampling, so data order is reproducible and
        # independent of any global-RNG consumption during forward/backward (e.g. dropout).
        self.data_gen = torch.Generator().manual_seed(cfg.seed)
        # Batch geometry. When tokens_per_step is set it is AUTHORITATIVE and grad_accum is
        # derived from the world size, so the global batch does not move when the GPU count
        # does. Without it we fall back to the configured grad_accum (single-process runs
        # and the in-memory test datasets).
        tps = getattr(cfg, "tokens_per_step", 0)
        if tps:
            self.seqs_per_step, self.grad_accum = dist_util.batch_geometry(
                tps, cfg.block_size, cfg.batch_size)
        else:
            self.seqs_per_step = cfg.grad_accum * cfg.batch_size * dist_util.world_size()
            self.grad_accum = cfg.grad_accum
        self.use_amp = self.device.startswith("cuda") and cfg.dtype == "bfloat16"
        self.step = 0
        # Last-step telemetry captured by train_step for side-effect-only TB logging.
        self.last_lr = cfg.lr
        self.last_grad_norm = 0.0
        # Probe NVML once so GPU telemetry (temp/util/power/clock) can log to TB. A missing
        # nvidia-ml-py or a probe failure -> memory-only telemetry, reported not swallowed
        # (telemetry must never be fatal to a multi-week run, nor vanish silently).
        self._gpu_nvml = False
        if self.device.startswith("cuda"):
            try:
                torch.cuda.temperature()
                self._gpu_nvml = True
            except Exception as e:  # noqa: BLE001 - report any NVML/import failure, don't crash
                print(f"GPU NVML telemetry unavailable (memory-only): {type(e).__name__}: {e}")

    def _autocast(self):
        if self.use_amp:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _next_batch(self, micro: int):
        """This rank's micro-batch for (step, micro).

        Uses the INDEXED sampler when the dataset offers one. That path is a pure function
        of (seed, global sequence index), so the set of sequences consumed at a given step
        is identical at every world size — proven at 1/2/4/8/16 in
        tests/test_migration_safe_loader.py. The legacy `get_batch` draws from a serialised
        torch.Generator whose state depends on how many processes have drawn from it, which
        cannot survive a change of world size.
        """
        cfg = self.cfg
        if hasattr(self.train_data, "get_batch_indexed"):
            idx = self.train_data.global_indices(
                self.step, micro, dist_util.rank(), dist_util.world_size(),
                cfg.batch_size, self.seqs_per_step)
            return self.train_data.get_batch_indexed(
                cfg.block_size, idx, cfg.seed, self.device)
        return self.train_data.get_batch(
            cfg.block_size, cfg.batch_size, self.device, self.data_gen)

    def train_step(self) -> float:
        cfg = self.cfg
        lr = get_lr(self.step, cfg)
        self.last_lr = lr
        for group in self.optimizer.param_groups:
            # lr_scale: 1.0 for AdamW groups; muon_lr/lr for Muon matrix groups, so both
            # follow one warmup+cosine shape at their own scale.
            group["lr"] = lr * group["lr_scale"]
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        accum = self.grad_accum
        for micro in range(accum):
            x, y = self._next_batch(micro)
            # DDP all-reduces during backward. Doing that on every micro-step would move
            # `accum` times more gradient than necessary; no_sync() defers it to the last
            # one, where the accumulated gradient is complete. The result is identical, the
            # traffic is 1/accum of it.
            last = micro == accum - 1
            ctx = self.ddp.no_sync() if (self.ddp is not None and not last) \
                else contextlib.nullcontext()
            with ctx:
                with self._autocast():
                    _, loss = self.model(x, y)
                    loss = loss / accum
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
        # The accumulated scalar is this RANK's mean, over its share of the step's
        # sequences — at world_size 2 that is half of them. The gradient is already global
        # (DDP averages it in backward); only the reported number is local, and leaving it
        # local makes two world sizes look like different runs when they are identical.
        return dist_util.all_reduce_mean(total_loss, self.device)

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
        # Rank 0 only. Eight ranks each writing a 16.6 GB checkpoint to the same path is a
        # corrupt file at best; the parameters are identical across ranks after DDP, so
        # rank 0's copy IS the checkpoint. The barrier keeps other ranks from racing ahead
        # into a step whose checkpoint is still being written.
        if not dist_util.is_main():
            dist_util.barrier()
            return
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
        dist_util.barrier()

    def _prune_checkpoints(self) -> None:
        keep = self.cfg.ckpt_keep
        if keep <= 0:
            return
        ckpts = sorted(
            Path(self.cfg.out_dir).glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1])
        )
        # Two-tier: protect the last `keep` (rolling crash-recovery window) AND every
        # milestone checkpoint (multiples of ckpt_milestone_interval — a permanent
        # training-trajectory record kept for later emergence/interpretability study).
        protected = set(ckpts[-keep:])
        milestone = self.cfg.ckpt_milestone_interval
        if milestone > 0:
            protected.update(p for p in ckpts if int(p.stem.split("_")[1]) % milestone == 0)
        for stale in ckpts:
            if stale not in protected:
                stale.unlink()

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.raw_model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        # lr_scale is config-derived, not learned state. MuonAdamW re-stamps its groups in
        # its own load_state_dict; AdamW checkpoints from before lr_scale existed (e.g. the
        # 1b run) restore groups without it, so stamp the uniform scale here.
        for group in self.optimizer.param_groups:
            if "lr_scale" not in group:
                group["lr_scale"] = 1.0
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
                    # GPU telemetry. If an NVML read fails mid-run, report once and drop to
                    # memory-only rather than crash the loop or silently stop logging.
                    try:
                        scalars = gpu_scalars(self.device, self._gpu_nvml)
                    except Exception as e:  # noqa: BLE001 - degrade telemetry, never kill training
                        self._gpu_nvml = False
                        scalars = gpu_scalars(self.device, False)
                        print(f"GPU NVML telemetry disabled: {type(e).__name__}: {e}")
                    for k, v in scalars.items():
                        writer.add_scalar(k, v, step)
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
