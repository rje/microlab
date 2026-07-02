> **Exercise — on `main`, no branch switching.** Implement the stub in
> `src/microlab/exercises/phase07_distributed.py`, then run
> `pytest tests/exercises/test_phase07_distributed.py -m exercise` to grade it.

# START HERE — distributed training (Phase 7)

How labs train models that don't fit on one GPU — and the phase where the 1B capstone
happens. One hand-write: `memory_budget` — params/grads/optimizer/activations per GPU
under data/tensor/pipeline parallelism and ZeRO stages 0-3. Every "can we afford to
train X" conversation in every lab starts with this arithmetic. Graded against the
oracle across a 7B config matrix.

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
