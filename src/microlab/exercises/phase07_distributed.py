"""Hand-write exercise (Phase 7): the per-GPU memory budget — the closed-form bookkeeping
every lab does before renting a cluster.

Fill in the ``NotImplementedError`` body so ``tests/exercises/test_phase07_distributed.py``
passes. Graded against ``microlab.distributed.reference.memory``. See
docs/hand-write/phase7-distributed.md.
"""

from __future__ import annotations


def memory_budget(n_params: int, n_layer: int, n_embd: int, block_size: int,
                  micro_batch: int, dp: int = 1, tp: int = 1, pp: int = 1,
                  zero_stage: int = 0, grad_checkpoint: bool = False,
                  dtype_bytes: int = 2) -> dict[str, float]:
    """Keys: params, grads, optimizer, activations, total (bytes per GPU).
    Model state / (tp*pp); ZeRO shards optimizer(>=1), grads(>=2), params(>=3) over dp.
    AdamW fp32 master = 12 bytes/param. Activations: (n_layer/pp) * micro_batch *
    block_size * n_embd * dtype_bytes * (1 if ckpt else 34) / tp."""
    raise NotImplementedError()
