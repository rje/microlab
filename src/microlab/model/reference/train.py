"""Training utilities for the reference GPT model.

Provides a full training loop with AMP/bf16 support on CUDA, and a simpler
`overfit_batch` helper used in tests to confirm the gradient flow is correct.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from microlab.data.reference.dataset import get_batch


@dataclass
class TrainConfig:
    """Configuration for a reference training run."""

    steps: int = 200
    batch_size: int = 16
    block_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_accum: int = 1
    device: str = "cuda"
    dtype: str = "bfloat16"
    log_every: int = 50
    seed: int = 1337


def _resolve_device(device: str) -> str:
    """Fall back to CPU if CUDA is requested but unavailable."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def train(model: torch.nn.Module, data: torch.Tensor, config: TrainConfig) -> dict:
    """Train `model` on a 1-D token tensor. Returns loss history + VRAM/throughput."""
    torch.manual_seed(config.seed)
    device = _resolve_device(config.device)
    model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.95)
    )
    use_amp = device.startswith("cuda") and config.dtype == "bfloat16"
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    )
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    gen = torch.Generator().manual_seed(config.seed)
    model.train()
    history: list[float] = []
    t0 = time.time()
    for _step in range(config.steps):
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(config.grad_accum):
            x, y = get_batch(data, config.block_size, config.batch_size, device, gen)
            with amp_ctx:
                _, loss = model(x, y)
                loss = loss / config.grad_accum
            loss.backward()
            step_loss += loss.item()
        opt.step()
        history.append(step_loss)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = max(time.time() - t0, 1e-9)
    tokens = config.steps * config.grad_accum * config.batch_size * config.block_size
    return {
        "final_loss": history[-1],
        "history": history,
        "tokens_per_sec": tokens / elapsed,
        "peak_vram_mb": (torch.cuda.max_memory_allocated() / 1e6)
        if device.startswith("cuda")
        else 0.0,
        "device": device,
    }


def overfit_batch(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int = 300,
    lr: float = 1e-3,
    device: str = "cpu",
) -> list[float]:
    """Train on ONE fixed batch; loss should collapse toward 0 if the loop is correct."""
    device = _resolve_device(device)
    model.to(device)
    x, y = x.to(device), y.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    losses: list[float] = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses
