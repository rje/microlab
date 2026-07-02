"""Reference per-GPU memory budget (Phase 7): where the bytes go when training with
data/tensor/pipeline parallelism and ZeRO. Closed-form and approximate on activations
(the 34-bytes-per-element multiplier is the standard no-recompute transformer estimate);
the cloud drills verify it against nvidia-smi reality."""

from __future__ import annotations

ACT_MULT_NO_CKPT = 34  # ~= attention + MLP intermediates per layer, bf16, no recompute


def memory_budget(n_params: int, n_layer: int, n_embd: int, block_size: int,
                  micro_batch: int, dp: int = 1, tp: int = 1, pp: int = 1,
                  zero_stage: int = 0, grad_checkpoint: bool = False,
                  dtype_bytes: int = 2) -> dict[str, float]:
    """Bytes per GPU for {params, grads, optimizer, activations, total}. Model state is
    split by tp*pp; ZeRO additionally shards optimizer (>=1), grads (>=2), params (>=3)
    across dp. Optimizer assumes AdamW with fp32 master weights (12 bytes/param)."""
    shard = tp * pp
    params = n_params * dtype_bytes / shard
    grads = n_params * dtype_bytes / shard
    optimizer = n_params * 12 / shard
    if zero_stage >= 1:
        optimizer /= dp
    if zero_stage >= 2:
        grads /= dp
    if zero_stage >= 3:
        params /= dp
    act_mult = 1 if grad_checkpoint else ACT_MULT_NO_CKPT
    activations = (n_layer / pp) * micro_batch * block_size * n_embd * dtype_bytes * act_mult / tp
    total = params + grads + optimizer + activations
    return {"params": params, "grads": grads, "optimizer": optimizer,
            "activations": activations, "total": total}
