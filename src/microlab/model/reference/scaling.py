"""Reference scaling tools (Phase 4): closed-form parameter counting, a training-FLOPs
estimate, a power-law fit, a model-family generator, and a sweep runner. The param
counter is graded against the real model's num_params(), so it must be exact."""

from __future__ import annotations

import math

import torch

from microlab.model.reference.gpt import GPT, GPTConfig
from microlab.model.reference.train import TrainConfig, train


def count_params(config: GPTConfig) -> dict[str, int]:
    """Exact parameter count for the reference GPT, broken down. Must equal
    GPT(config).num_params(). Token embedding is tied with the LM head (counted once);
    LayerNorms always contribute weight+bias; the `bias` flag only affects Linears."""
    V, C, L, Bk = config.vocab_size, config.n_embd, config.n_layer, config.block_size
    b = 1 if config.bias else 0
    token_emb = V * C        # wte (tied with lm_head)
    pos_emb = Bk * C         # wpe
    attn = (3 * C * C + 3 * C * b) + (C * C + C * b)          # c_attn + c_proj
    mlp = (4 * C * C + 4 * C * b) + (4 * C * C + C * b)       # c_fc + c_proj
    norms = 2 * C + 2 * C                                     # ln_1 + ln_2 (weight+bias)
    per_block = attn + mlp + norms
    final_norm = 2 * C                                        # ln_f
    non_embedding = L * per_block + final_norm
    total = token_emb + pos_emb + non_embedding
    return {
        "token_emb": token_emb,
        "pos_emb": pos_emb,
        "per_block": per_block,
        "non_embedding": non_embedding,
        "total": total,
    }


def training_flops_per_token(config: GPTConfig) -> int:
    """Approx training FLOPs per token = 6 * N_non_embedding (Kaplan/Chinchilla: ~2N
    forward + ~4N backward). Attention's quadratic term is ignored (small at this scale)."""
    return 6 * count_params(config)["non_embedding"]


def training_flops(config: GPTConfig, n_tokens: int) -> int:
    return training_flops_per_token(config) * n_tokens


def fit_scaling_law(params: list[int], losses: list[float]) -> tuple[float, float]:
    """Fit L = A * N^(-alpha) by log-log least squares. Returns (A, alpha)."""
    import numpy as np

    log_n = np.log(np.asarray(params, dtype=float))
    log_l = np.log(np.asarray(losses, dtype=float))
    slope, intercept = np.polyfit(log_n, log_l, 1)  # log L = intercept + slope*log N
    return float(math.exp(intercept)), float(-slope)


def model_family(
    sizes: list[int], vocab_size: int = 512, block_size: int = 128, head_dim: int = 32
) -> list[GPTConfig]:
    """GPTConfigs of increasing width/depth at a fixed aspect ratio. `sizes` are n_embd
    values; n_head = n_embd // head_dim; depth grows with width."""
    configs = []
    for c in sizes:
        configs.append(
            GPTConfig(
                vocab_size=vocab_size,
                block_size=block_size,
                n_layer=max(2, c // 64),
                n_head=max(1, c // head_dim),
                n_embd=c,
            )
        )
    return configs


def run_scaling_sweep(data: torch.Tensor, sizes: list[int], train_cfg: TrainConfig,
                      val_data: torch.Tensor | None = None, vocab_size: int = 512) -> dict:
    """Train each family member on the same data, collect (params, loss), fit the power
    law. Fits on held-out `val_data` loss when provided (the meaningful signal), else on
    train loss. GPU-friendly: pass a CUDA TrainConfig."""
    points = []
    for cfg in model_family(sizes, vocab_size=vocab_size, block_size=train_cfg.block_size):
        torch.manual_seed(train_cfg.seed)
        model = GPT(cfg)
        stats = train(model, data, train_cfg, val_data=val_data)
        points.append({"n_embd": cfg.n_embd, "params": model.num_params(),
                       "loss": stats["final_loss"], "val_loss": stats["val_loss"]})
    fit_on = "val" if points and points[0]["val_loss"] is not None else "train"
    fit_losses = [(p["val_loss"] if fit_on == "val" else p["loss"]) for p in points]
    a, alpha = fit_scaling_law([p["params"] for p in points], fit_losses)
    return {"points": points, "A": a, "alpha": alpha, "fit_on": fit_on}
