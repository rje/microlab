"""Distributed helpers: process-group lifecycle, rank identity, and batch geometry.

The design constraint that shapes this file: **the global batch must be invariant to world
size**, so a run started on one GPU and finished on four is the SAME run rather than a
similar one. That is why the batch is specified in TOKENS PER STEP and `grad_accum` is
derived, instead of `grad_accum` being a config knob:

    seqs_per_step = tokens_per_step / block_size          (independent of world size)
    grad_accum    = seqs_per_step / (world_size * batch_size)

At tokens_per_step=524,288 and block_size=32,768 that is 16 sequences per step:
world_size 1 -> grad_accum 16; world_size 4 -> grad_accum 4; world_size 8 -> grad_accum 2.
Same sixteen sequences, same gradient, distributed differently.

Setting grad_accum by hand across a world-size change would move the effective batch by the
world-size factor and silently invalidate both the LR schedule and the token accounting —
the run would keep training and the loss curve would look plausible.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    """True when launched under torchrun (or any launcher setting the env contract)."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def rank() -> int:
    return int(os.environ.get("RANK", 0))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def is_main() -> bool:
    """Rank 0. Checkpointing, logging and eval printing are gated on this."""
    return rank() == 0


def setup(device: str = "cuda", backend: str | None = None) -> str:
    """Initialise the process group and bind this rank to its GPU.

    Returns the device string this rank should use. Defaults to NCCL on CUDA and gloo
    otherwise — gloo is what makes a 2-rank correctness test possible on a single-GPU box.
    """
    if not is_distributed():
        return device
    if backend is None:
        backend = "nccl" if device.startswith("cuda") and torch.cuda.is_available() \
            else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    if backend == "nccl":
        torch.cuda.set_device(local_rank())
        return f"cuda:{local_rank()}"
    return device


def teardown() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def all_reduce_mean(value: float, device: str) -> float:
    """Average a python scalar across ranks — for logging a global loss, not a local one.

    Reporting rank 0's loss alone would understate variance and make two runs at different
    world sizes look different when they are not.
    """
    if not dist.is_initialized():
        return value
    t = torch.tensor([value], dtype=torch.float64,
                     device=device if device.startswith("cuda") else "cpu")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / world_size())


def batch_geometry(tokens_per_step: int, block_size: int, batch_size: int,
                   ws: int | None = None) -> tuple[int, int]:
    """(seqs_per_step, grad_accum) for this world size, or raise if it does not divide.

    Refusing an indivisible layout is deliberate. Silently rounding grad_accum would change
    the effective batch on exactly the runs where it matters most — a migration to a
    different GPU count — and nothing downstream would notice.
    """
    ws = world_size() if ws is None else ws
    if tokens_per_step % block_size:
        raise ValueError(
            f"tokens_per_step {tokens_per_step:,} is not divisible by block_size "
            f"{block_size:,}")
    seqs = tokens_per_step // block_size
    per_rank = ws * batch_size
    if seqs % per_rank:
        raise ValueError(
            f"{seqs} sequences/step does not divide across world_size {ws} x batch_size "
            f"{batch_size}. Pick tokens_per_step so that "
            f"tokens_per_step/block_size is a multiple of world_size*batch_size.")
    return seqs, seqs // per_rank


def barrier() -> None:
    """No-op when not distributed, so callers need no branch."""
    if dist.is_initialized():
        dist.barrier()
