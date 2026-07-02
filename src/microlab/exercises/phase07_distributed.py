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
    """Per-GPU training memory budget. Returns a dict with keys ``params``, ``grads``,
    ``optimizer``, ``activations``, ``total`` (bytes per GPU). Derive the byte-accounting —
    the model-state bytes, the ZeRO sharding tiers, the AdamW fp32 master-copy cost, and the
    activation term with/without gradient checkpointing — from
    docs/hand-write/phase7-distributed.md. Graded against the reference across a 7B config
    matrix."""
    raise NotImplementedError(
        "derive the per-GPU memory budget — see docs/hand-write/phase7-distributed.md"
    )
