# Running the 1B pretrain on hosted hardware

Sketch written 2026-08-01, while the frontier validation runs locally. Numbers are measured
where marked; provider prices need re-checking at booking time.

## The economics, and why renting is obviously right

| | our RTX 6000 Ada | 8x H100 spot | 8x A100 spot |
|---|---|---|---|
| 1.56e20 FLOPs (1.04B params x 25B tokens) | **18-33 days** | **~12 h** | ~27 h |
| cost | electricity (~$1.50/day) | **~$193** | ~$217 |

The local figure is measured, not modelled: our token-matched benchmark puts dense at 8,811
tok/s at 4k context, giving 32.8 days for 25B tokens, or ~18 days optimistically with
compile and no gradient checkpointing.

**The dominant cost risk is not the GPU price. It is paying for idle GPUs** — while data
downloads, while `max-autotune` compiles, while someone debugs a config, or while a run
that was going to fail anyway keeps going. Every item below is aimed at that.

## Blocking gap: we have never run multi-GPU

`grep` finds no `DistributedDataParallel`, no FSDP, no `dist.init_process_group`, no
`torchrun` entry point. The trainer is single-process only. **This is the single largest
risk in the plan** — the first multi-GPU run would otherwise happen on a paid clock.

**DDP is sufficient; FSDP is not needed.** At 1.04B params a full checkpoint is ~16.6 GB
(2.08 params bf16 + 4.16 fp32 copy + 8.32 AdamW m/v + 4.16 Muon momentum), which fits an
80 GB card many times over. FSDP's sharding buys nothing here and adds a large new surface.

Work required:
1. `torchrun` entry point; `dist.init_process_group("nccl")`; wrap the model in DDP.
2. Shard the data loader by rank so the 8 processes do not all read the same batches.
3. Rank-0-only checkpointing and logging.
4. Muon needs checking under DDP — it orthogonalises per-parameter-matrix via Newton-Schulz,
   and gradients must be all-reduced BEFORE that step, not after. Getting this backwards
   would silently train a different algorithm.
5. Verify locally with `torchrun --nproc_per_node=1`, which exercises the whole distributed
   code path with one GPU. It cannot catch real multi-rank bugs (deadlocks, uneven sharding).

## The shakedown run: $16 to de-risk $200

Before the real run, rent the target instance for **one hour** and check:
- all 8 ranks initialise and NCCL all-reduce works
- measured tokens/sec, extrapolated to a total cost, against the local prediction
- MFU: if we are at 30% instead of 50% we are paying ~1.6x more than the estimate
- a checkpoint writes to persistent storage and a resume from it works
- deliberately kill a rank and confirm the run dies cleanly rather than hanging

That last one matters for spot: a hung job burns money silently, where a crashed job
restarts.

## Storage and data staging

Corpus footprint today, and it grows with the remaining slices:

| slice | size |
|---|---|
| code-repo-32k | 52 GB |
| web-49k | 15 GB |
| math-49k | 5.7 GB |
| arxiv + commits + markdown | ~10 GB (in progress) |
| **mixed corpus (final)** | **~55 GB** (27B tokens x 2 bytes) |

**Stage to persistent/network storage, never instance-local.** Spot instances vanish; the
data must outlive them. Upload once, mount for every subsequent run.

Options, in the order I would try them:
1. **Provider network volume** (Lambda persistent filesystem, RunPod network volume). Fast,
   attached at boot, survives preemption. Costs a few $/month.
2. **HuggingFace Hub private dataset.** Free, good bandwidth, already in our stack, and
   gives versioning for free. Slower to pull on each new instance (~55 GB), so pair it with
   a network volume rather than re-downloading per run.
3. **S3/R2/GCS.** Most flexible, needs credentials plumbing.

**Pull the data before the GPUs start billing where possible** — some providers bill from
instance start, so downloading 55 GB at instance boot is dead money. A network volume
prepared in advance avoids that entirely.

## Checkpoints

~**16.6 GB each**. Two separate concerns:

**Preemption insurance.** Spot means losing everything since the last checkpoint. Cadence
should be ~15-30 minutes of work. At a plausible 2M-token global batch, 21B tokens is
~10,500 steps over ~12 h, so ~4 s/step — checkpoint every ~400 steps.

**Research trajectory.** Separate from the rolling recovery window. The trainer already
supports this split (`ckpt_interval` + `ckpt_keep` for rolling, `ckpt_milestone_interval`
for permanent), which was built for the 1B. Keep milestones at e.g. every 2,000 steps.

**Write checkpoints to the network volume, not instance disk**, for the same reason as the
data. Retrieval afterwards is then just a download from the volume or a push to the Hub.

## Pre-flight checklist for the rented run

Run all of this BEFORE the instance is billing:

- [ ] `scripts/preflight_lane.py` clean on the 1B config
- [ ] parity review re-run against the frozen 1B config (`sota-parity-code-specialist.md`)
- [ ] the mixed corpus is built, and its document-span check confirms 32k windows are
      repo-coherent for the code slice
- [ ] FIM transform implemented and round-trip tested
- [ ] DDP verified under `torchrun --nproc_per_node=1`
- [ ] a written prediction of the expected loss curve, so the outcome is falsifiable
- [ ] a hard wall-clock budget in the launch script that kills the job if exceeded

## What could go wrong and what it would cost

| failure | cost | mitigation |
|---|---|---|
| spot preemption | work since last checkpoint | 400-step cadence to a network volume |
| a rank hangs (NCCL) | silent, full-price burn | shakedown test + a wall-clock watchdog |
| MFU far below estimate | 1.5-2x the projected bill | measure in the shakedown, not the real run |
| data not staged, downloads on GPU time | ~1 h x $16 | network volume prepared in advance |
| compile every restart | ~15 min x $16/h per restart | persist the inductor cache to the volume |
| config wrong (the 1B's MHA class of error) | the whole run | preflight gate + parity review |
| Muon wrong under DDP | trains a different algorithm, invisibly | verify the all-reduce ordering explicitly |

## Realistic budget

~$193 for the run + ~$16 shakedown + a contingency for one preemption/restart cycle
=> **~$250**, with the wall-clock watchdog as the hard stop.
