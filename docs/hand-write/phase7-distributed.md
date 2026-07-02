> **Exercise — on `main`, no branch switching.** Implement the stub in
> `src/microlab/exercises/phase07_distributed.py`, then run
> `pytest tests/exercises/test_phase07_distributed.py -m exercise` to grade it.

# START HERE — distributed training (Phase 7)

How labs train models that don't fit on one GPU — and the phase where the 1B capstone
happens. One hand-write: `memory_budget` — params/grads/optimizer/activations per GPU
under data/tensor/pipeline parallelism and ZeRO stages 0-3. Every "can we afford to
train X" conversation in every lab starts with this arithmetic. Graded against the
oracle across a 7B config matrix.

## The arithmetic (derive it — don't transcribe it)

`memory_budget` returns bytes per GPU under the keys `params`, `grads`, `optimizer`,
`activations`, `total`. Derive each term from first principles; the readings below are where
the constants come from.

- **Model state** is three tensors sized by the parameter count: the weights and the
  gradients each cost `dtype_bytes` per param (bf16 → 2), and the AdamW optimizer state costs
  **12 bytes/param** — an fp32 master copy of the weights plus the fp32 first and second
  moments (4 + 4 + 4). This is the "where the 12 bytes/param go" the ZeRO paper opens with.
- **Model-parallel split:** tensor and pipeline parallelism each shard the model state, so
  params / grads / optimizer are divided by `tp * pp` (each rank owns its slice of the layers
  and the matmuls). Megatron-LM is the tensor-parallel half.
- **ZeRO tiers:** data parallelism replicates the model state by default; ZeRO shards it
  across the `dp` group instead, in stages — stage ≥1 shards the **optimizer** state, ≥2 also
  the **gradients**, ≥3 also the **params**. Divide each sharded term by `dp`.
- **Activations** are the memory that scales with the batch, not the parameters:
  `(n_layer / pp) · micro_batch · block_size · n_embd · dtype_bytes · M / tp`, where `M` is
  the per-element multiplier — ~**34** for a transformer layer with no recompute (attention +
  MLP intermediates, bf16), or **1** with `grad_checkpoint=True`, where you keep only the
  block-boundary activations and recompute the rest in the backward pass (the ~30× drop you
  measure in rung 1).
- **`total`** is their sum.

## The three rungs

1. **Local (free):** set `grad_checkpoint=True` in a config and watch VRAM drop ~30x on
   activations while tokens/sec dips ~25%; set `compile=True` and watch tokens/sec rise.
   Measure both on the 150M config; record the table.
2. **Cloud drills (~$25-50):** `ops/lambda-distributed.md`. DDP the 150M across 1/2/4
   GPUs, measure scaling efficiency; FSDP the 1B config and check your memory_budget()
   prediction against nvidia-smi. Your closed-form exercise meets reality here.
3. **The 1B capstone:** venue decided by the vendor spike (runbook step 0). LR comes from
   Phase 4's muP transfer, the fit-check from your memory budget, the restart durability
   from the systemd + resume infrastructure already proven on the 150M run.

## Readings

Megatron-LM (tensor parallelism: split the matmuls), ZeRO (shard the optimizer states —
where the 12 bytes/param actually go). Both in the console.
