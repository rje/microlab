"""Run configuration for the production Trainer (Phase-2 real-scale). One dataclass so a
150M and a 1B run are just different config values."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunConfig:
    # model (VariantConfig fields)
    vocab_size: int = 32000
    block_size: int = 512
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    norm: str = "rms"
    pos: str = "rope"
    mlp: str = "swiglu"
    # optim / schedule
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 100
    max_steps: int = 1000
    lr_decay_steps: int = 1000
    # data / io
    batch_size: int = 16
    grad_accum: int = 1
    eval_interval: int = 250
    eval_iters: int = 50
    ckpt_interval: int = 500
    ckpt_keep: int = 3  # 0 disables pruning; only the last N ckpt_*.pt files are kept
    grad_checkpoint: bool = False  # recompute activations backward: ~30x less act memory
    compile: bool = False          # torch.compile the model (CUDA; first step compiles)
    log_interval: int = 50
    out_dir: str = "runs/pretrain"
    device: str = "cuda"
    dtype: str = "bfloat16"
    seed: int = 1337
