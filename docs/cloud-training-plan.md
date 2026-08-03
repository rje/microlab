# Training the 1B on rented hardware — plan, providers, and the validation ladder

Written 2026-08-03, after the local run was stopped at step 75 and
`docs/train-config-bench.json` established that ~46 days is close to what the RTX 6000 Ada
does for this model (46% MFU; the whole spread across working config variants is 10.5%).

**This supersedes the budget in `docs/hosted-training-plan.md`, which said ~$193 + $16
shakedown = ~$250. That was optimistic by 2-3x.** It assumed ~90% MFU implicitly and 12
hours of wall-clock. Recomputed from our own measured throughput, the run is ~20 hours and
the realistic all-in is **$300-700 depending on provider**. The rest of that document
(checkpoint cadence, failure table, staging discipline) still stands.

## 1. What the run actually costs

Training FLOPs at 6ND (an 80 GB H100 does not need gradient checkpointing, unlike our
48 GB card, so the recompute term disappears): **1.28e20**.

| aggregate MFU | 8xH100 throughput | wall-clock |
|---|---|---|
| 35% (pessimistic, first run) | 1.39 PFLOP/s | 25.6 h |
| 45% (realistic) | 1.78 PFLOP/s | 19.9 h |
| 55% (good) | 2.18 PFLOP/s | 16.3 h |

Budget on **20 hours** and treat anything under 26 as success.

## 2. Providers

| | $/GPU-h | $/h (x8) | 20 h run | NVLink | interruption | verdict |
|---|---|---|---|---|---|---|
| **Lambda 1-Click Cluster** | $3.99 | $31.92 | **$638** | NVSwitch, guaranteed | on-demand, not preempted | safest, dearest |
| **RunPod Secure Cloud** | $2.99 | $23.92 | **$478** | SXM nodes have it | rare, but [227+ outages/9mo reported](https://www.spheron.network/blog/runpod-vs-vastai-2026/) | best balance |
| **RunPod Community** | $2.69 | $21.52 | $430 | host-dependent | host-dependent | thin saving, real variance |
| **Vast.ai verified DC** | $1.50 | $12.00 | **$240** | **varies by host** | interruptible | cheapest, most work |

### The tradeoff that actually matters

Conventional advice is that only Lambda does real multi-GPU because it guarantees
NVLink/NVSwitch. **For our job that advice is probably over-conservative, and it is worth
checking rather than paying $400 to avoid.** DDP on a 1B model all-reduces
1.013e9 x 2 bytes = **2.03 GB of gradients per step**. Ring all-reduce moves ~2x that per
GPU, so ~4 GB per step. At ~20 h / 40,000 steps that is ~1.8 s/step of compute budget:

- over NVLink (~450 GB/s): ~9 ms, i.e. 0.5% overhead
- over PCIe 4.0 x16 (~25 GB/s effective): ~160 ms, i.e. **~9% overhead**

9% of a $478 run is $43. So a non-NVLink host is a real but survivable tax, not a
disqualifier — which is exactly the kind of claim the shakedown should settle with a
measurement instead of an argument. **Measure all-reduce bandwidth in the shakedown and
pick the provider on the number.**

Vast.ai's catch is not bandwidth, it is variance: it is a marketplace of individual hosts,
so disk speed, network egress rate and host reliability differ per listing, and an
interrupted job needs the resume path to be flawless. Our resume path is checkpoint-based
and the data loader is deterministic from `(seed, step)`, so we are unusually well placed
to absorb that — but only after the resume equivalence test below passes.

**Recommendation: RunPod Secure Cloud for the real run** ($478), with the shakedown run on
whichever of RunPod/Vast is available. Lambda only if the shakedown shows we are
interconnect-bound and Vast/RunPod cannot supply NVLink.

## 3. Storage — already costed, needs an account

**Backblaze B2**, not Cloudflare R2. This reverses the choice in
`docs/hosted-training-plan.md`, on two facts found after it was written:

1. [Vast's cloud sync](https://docs.vast.ai/instances/cloud-sync) supports **S3, Backblaze,
   Google Drive and Dropbox — not R2.** R2 would still work via `rclone` (it is
   S3-compatible and Vast cannot prevent it), but we would give up the native integration
   for nothing.
2. B2 is cheaper at our shape.

| | storage/mo | egress/mo | total/mo | Vast native sync |
|---|---|---|---|---|
| Cloudflare R2 | $3.15 | $0.00 | $3.15 | no |
| **Backblaze B2** | **$1.53** | **$0.00** | **$1.53** | **yes** |
| AWS S3 | $5.06 | $19.44 | $24.50 | yes |

R2's one advantage was uncapped free egress against B2's cap of 3x stored data. At 220 GB
stored that cap is 660 GB/month and our actual use is ~216 GB — **33% of it.** The
constraint does not bind, so it does not justify the pricier, less-integrated option.

Reliability, for the record, since it was asked: R2 offers eleven nines of durability and a
**99.9% availability SLA**, and has had real outages — 2025-03-21 lost 100% of writes and
~35% of reads globally for 1h07m. B2 is comparable. Either is fine here, because the corpus
is on instance disk before training starts and checkpoint uploads retry; the mitigation
that matters is keeping a LOCAL rolling checkpoint alongside the remote one, so an outage
coinciding with a preemption does not cost work.

Account steps (one-time, ~10 min):
1. Backblaze account -> B2 Cloud Storage (card on file; 10 GB free).
2. Create buckets `microlab-corpus` and `microlab-ckpt`, both **private**.
3. App Keys -> **Add a New Application Key**, scoped to those buckets only, read+write.
   Vast's own guidance is exactly this: a dedicated key, never a full-access one.
4. Note the **S3 endpoint** from the bucket page, e.g.
   `https://s3.us-west-004.backblazeb2.com`.
5. `scripts/b2_sync.py up --local data/shards/mix-v1 --bucket microlab-corpus
   --prefix mix-v1` — 40 GB, ~1 h at 100 Mbit up. Resumable: objects whose remote size
   already matches are skipped.

Secrets go in `~/.config/microlab/b2.env` (mode 600, enforced), never in the repo, and
never on the command line — argv is world-readable via /proc. The launch script reads them
from the environment.

## 4. What has to be built

Ordered by dependency. Items 1-3 are the blocking work; 4-6 are the money-savers.

| # | item | why | est |
|---|---|---|---|
| 1 | **`torchrun` entry + DDP wrap** | we have never run multi-GPU; single largest risk | 0.5 d |
| 2 | **Rank-sharded loader wiring** | `sequence_at`/`global_indices` exist and are tested at world sizes 1-16; the trainer does not call them yet | 0.5 d |
| 3 | **Rank-0 checkpointing + logging** | 8 ranks writing 16.6 GB each would be a disaster | 0.25 d |
| 4 | **Muon under DDP** | it orthogonalises per matrix via Newton-Schulz; gradients must be all-reduced BEFORE that step. Getting it backwards trains a different algorithm, invisibly | 0.5 d |
| 5 | **`scripts/cloud_launch.sh`** | pull corpus from B2 -> start torchrun -> async rank-sharded checkpoint upload -> wall-clock watchdog | 0.5 d |
| 6 | **`scripts/b2_sync.py`** | upload/download with resume, integrity check against the manifest | 0.25 d |

~2.5 days. Everything downstream of the data loader is already done and tested: the mix is
built, FIM is in, the batch is world-size invariant, and the preflight gate covers duration,
memory, trainability, decode and arm parity.

### Muon under DDP — the one that can fail silently

DDP all-reduces gradients in the backward pass, so by the time `optimizer.step()` runs the
gradients are already averaged. Muon then orthogonalises. That ordering is correct. The
failure mode is if any part of the Muon update is computed from a rank's LOCAL gradient
before the all-reduce, in which case each rank orthogonalises a different matrix and the
result is not Muon at all — and the loss curve would look plausible. **Validation: run
world_size=2 and world_size=1 with the same seed over the same global batch and assert the
parameters match to numerical tolerance after N steps.** That is L2 below.

## 5. The validation ladder — how we avoid burning money

Each rung is cheap and kills the plan before the next rung costs more. **Do not skip a rung
because the previous one passed.**

### L0 — free, local, no GPU rental
- `preflight_lane.py` clean on the final config (already passing: 1013M params, 1.03x
  Chinchilla, decode verified).
- `torchrun --nproc_per_node=1` completes 20 steps. Exercises the entire distributed code
  path with one GPU: process group init, DDP wrap, rank-sharded loader, rank-0 checkpoint.
  It cannot catch real multi-rank bugs, and must not be mistaken for evidence that it works.

### L1 — free, local, CPU: multi-rank logic without a second GPU
- `torchrun --nproc_per_node=2` with the **gloo** backend on a toy config (2 layers, tiny
  vocab). Proves the collective calls, sharding arithmetic and rank-0 gating are right
  without needing two cards. NCCL-specific problems will not surface here; everything else
  will.

### L2 — free, local: THE equivalence test
The single most valuable check we own, and it exists because the loader was built for it.
- Train 20 steps at `world_size=1, grad_accum=16`. Record loss per step and the final
  parameter hash.
- Train 20 steps at `world_size=2, grad_accum=8` (both ranks on the one GPU via gloo, or
  sequentially simulated).
- **Assert the loss curves match to numerical tolerance.** The global batch is proven
  identical at world sizes 1-16 by `tests/test_migration_safe_loader.py`; this extends that
  from *the same data* to *the same optimizer trajectory*, and it is what makes the cloud
  run the same experiment rather than a similar one.
- Also catches the Muon-ordering bug in item 4.

### L3 — ~$25-32, one hour, 8 GPUs: the shakedown
Rent the real target for **one hour** and check, in this order, stopping at the first failure:
1. all 8 ranks init, NCCL all-reduce completes
2. **measured all-reduce bandwidth** -> decides NVLink-vs-PCIe and therefore the provider
3. corpus pull from B2 to local NVMe: wall-clock and MB/s (the one number in the storage
   plan we have never observed)
4. **tokens/sec and MFU**, extrapolated to a total cost and compared against the 20 h
   prediction above. If MFU is 30% not 45%, the run costs 1.5x more and that is knowable
   here for $30 rather than after the fact
5. checkpoint writes to B2 and a resume from it produces the same loss
6. deliberately `kill -9` one rank and confirm the job **dies** rather than hanging. A hung
   job burns money silently; a crashed one restarts.

Gate: proceed only if MFU >= 30% and resume is bit-consistent.

### L4 — the run
- Hard wall-clock watchdog in the launch script that kills the job at 1.5x the predicted
  time. Non-negotiable: this is the difference between a $478 run and an open-ended bill.
- Checkpoint every ~15-30 min of work to B2 (async, rank-sharded).
- A **written prediction of the expected loss curve at 2B/5B/10B/21B tokens** before
  launch, from the frontier-32k run and the original 1B, so "is this working" is falsifiable
  at hour 2 instead of hour 20.
- `scripts/eval_suite.py` against each milestone as it lands.

## 6. Budget

| item | cost |
|---|---|
| B2 storage, one month | $2 |
| L3 shakedown (1 h, 8xH100) | $24-32 |
| Run at 20 h, RunPod Secure | $478 |
| Contingency: one preemption + restart (~3 h) | $72 |
| **total** | **~$590** |

On Vast.ai verified the same shape is **~$300**; on Lambda **~$780**. The spread is real
money and the shakedown is what tells us whether the cheap option is safe for our job.

Local remains ~46 days and ~$69 of electricity. **The honest framing is $590 to buy back six
weeks of the card**, not "cloud is cheaper."

## 7. Decision points for the owner

1. **Provider**: recommend RunPod Secure for the real run; shakedown wherever is available.
2. **Do we run the shakedown on a different provider than the real run?** Recommend yes if
   Vast is much cheaper at shakedown time — the numbers transfer.
3. **Token budget**: hold at 21B (1.03x Chinchilla). Cutting to 10B halves the bill but is
   0.49x Chinchilla, below the 0.30x ratio that has already inverted verdicts on us.
4. **Who pushes the button on L4** — the watchdog bounds the downside, but the go/no-go
   after reading L3's MFU number should be explicit.
