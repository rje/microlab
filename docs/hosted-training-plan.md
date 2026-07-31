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

### What we are actually storing

Slices on disk today (`du` on `data/shards`):

| slice | size |
|---|---|
| code-repo-32k | 54.8 GB |
| web-49k | 16.0 GB |
| math-49k | 6.0 GB |
| arxiv-49k | 14.8 GB |
| commits-49k | <0.1 GB |
| **raw slices total** | **91.6 GB** |

The raw slices stay local. What ships is the **built mix — 27B tokens x 2 bytes = 54 GB**,
because the mix builder samples from the slices at the target ratios rather than shipping
all of them. Plus checkpoints at **16.6 GB each**; keeping ten is 166 GB.

**Steady-state bucket: ~220 GB. Egress: ~54 GB per training run.**

### Recommendation: Cloudflare R2 as the canonical store, instance-local NVMe as the hot path

Priced against the three real options, at our 220 GB / 54-GB-per-run shape and four runs a
month (216 GB egress):

| | storage/mo | egress/mo | **total/mo** | note |
|---|---|---|---|---|
| **Cloudflare R2** | $3.15 | **$0.00** | **$3.15** | $0.015/GB-mo, egress free unconditionally, 10 GB free tier |
| Backblaze B2 | $1.53 | $0.00 | $1.53 | $6.95/TB-mo; egress free to 3x stored (660 GB for us) |
| AWS S3 Standard | $5.06 | $19.44 | $24.50 | $0.023/GB-mo + **$0.09/GB out** |
| RunPod network volume | $15.40 | n/a | $15.40 | $0.07/GB-mo under 1 TB; region-pinned |

**S3 is the one to avoid, and egress is the entire reason** — $48.60 of pure transfer cost
over a ten-run project, on a project whose GPU budget is ~$250. R2 and B2 both make that
number zero.

Between R2 and B2 the difference is $1.62/month, which is noise. **Take R2**, for two
reasons that are not about price:
- Its free egress has **no cap to reason about**. B2's is free up to 3x average monthly
  storage, so a burst of runs against a small bucket could cross it. R2 has no such edge.
- **It makes co-location a non-issue.** With S3/GCS you must place the bucket in the same
  region as the GPUs or pay cross-cloud egress, which constrains which provider and region
  you can rent from — exactly the wrong constraint when chasing spot capacity. With free
  egress, the bucket is equidistant from every provider and we rent wherever it is cheapest.

Both are S3-compatible, so `boto3`/`rclone`/`s5cmd` work unchanged and the choice is
reversible for the cost of one copy.

### Why NOT a provider network volume

This reverses the earlier draft of this document, which said "stage to network storage,
never instance-local." That was wrong on the arithmetic.

A network volume costs $15.40/month (RunPod, 220 GB) and **pins the job to one
datacenter** — you can only start pods in the volume's region, which shrinks the spot pool
we are specifically trying to shop across. What it buys is avoiding a re-pull after
preemption. Priced:

- 54 GB pull at 10 Gbps = **0.7 min = $0.19** of GPU time at $16/h.
- 54 GB pull at 1 Gbps = 7.2 min = $1.92.

At $0.19–1.92 per restart, the $15.40/month volume only pays for itself somewhere north of
eight restarts a month, and it costs us region flexibility to get there. Lambda's 8xH100
instances ship 22 TiB of local SSD; there is no capacity argument either.

**So: pull the mix from R2 to instance-local NVMe at job start.** The corpus is
re-pullable, so losing it with the instance costs a minute, not a day.

### Checkpoints: the one thing that genuinely must leave the instance

Instance-local disk dies with a preempted spot instance, so rolling recovery checkpoints on
local NVMe are worthless as preemption insurance — that is the exact failure they exist to
cover. **Recovery checkpoints go to R2.**

Two consequences to design for, both of which need measuring in the shakedown:

1. **The upload must be asynchronous.** 16.6 GB at ~1 Gbps is ~133 s. At a 400-step cadence
   (~27 min) a synchronous write is ~8% of wall-clock — roughly **$16 on a $200 run**, paid
   for doing nothing. Overlapping the upload with the next steps makes it free.
2. **Shard the upload across ranks.** Each of the 8 ranks uploading its own slice gives 8
   parallel streams, which object storage serves far better than one.

Ingress to R2 is free and operations are negligible: 16.6 GB in 100 MB multipart chunks is
~166 Class A ops, so a full 12 h run's ~27 checkpoints costs about **$0.02** in requests.

### HuggingFace Hub: the durable public copy, not the working store

Push the built mix and its **attribution manifest** to a Hub dataset repo. This is not the
training data path — it is the archival and provenance copy, and the attribution manifest is
a hard shipping requirement under the Stack v1 dedup agreement. Free, versioned, and it
survives us deleting the R2 bucket. Do not put 27-minute checkpoints there.

### Concretely

1. `rclone`/`s5cmd` the built mix (54 GB) to `r2://microlab-corpus/mix-v1/` once.
2. Job start pulls it to instance NVMe with parallel workers; measure the wall-clock in the
   shakedown, since it is the one number here we have not observed.
3. Rolling checkpoints upload async, rank-sharded, to `r2://microlab-ckpt/<run>/`.
4. Milestones additionally get pushed to the Hub at the end of the run.
5. Prune rolling checkpoints to `ckpt_keep` on R2 too, or the 220 GB assumption drifts —
   27 unpruned checkpoints is 448 GB, which would triple the storage line.

## Checkpoints

~**16.6 GB each**. Two separate concerns:

**Preemption insurance.** Spot means losing everything since the last checkpoint. Cadence
should be ~15-30 minutes of work. At a plausible 2M-token global batch, 21B tokens is
~10,500 steps over ~12 h, so ~4 s/step — checkpoint every ~400 steps.

**Research trajectory.** Separate from the rolling recovery window. The trainer already
supports this split (`ckpt_interval` + `ckpt_keep` for rolling, `ckpt_milestone_interval`
for permanent), which was built for the 1B. Keep milestones at e.g. every 2,000 steps.

**Write checkpoints to R2, not instance disk** — see the storage section. Local disk dies
with a preempted instance, which defeats the entire purpose of a recovery checkpoint.
Retrieval afterwards is a download from the bucket.

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
| spot preemption | work since last checkpoint | 400-step cadence, uploaded to R2 |
| a rank hangs (NCCL) | silent, full-price burn | shakedown test + a wall-clock watchdog |
| MFU far below estimate | 1.5-2x the projected bill | measure in the shakedown, not the real run |
| synchronous checkpoint upload | ~8% of wall-clock, ~$16/run | async, rank-sharded upload |
| slow pull from object storage | $0.19-1.92 per job start | parallel workers; measure in the shakedown |
| unpruned checkpoints on R2 | 448 GB not 220 GB, 3x the storage line | apply `ckpt_keep` to the bucket, not just disk |
| compile every restart | ~15 min x $16/h per restart | persist the inductor cache to R2 |
| config wrong (the 1B's MHA class of error) | the whole run | preflight gate + parity review |
| Muon wrong under DDP | trains a different algorithm, invisibly | verify the all-reduce ordering explicitly |

## Realistic budget

~$193 for the run + ~$16 shakedown + a contingency for one preemption/restart cycle
=> **~$250**, with the wall-clock watchdog as the hard stop.
