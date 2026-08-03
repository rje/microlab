# Plan for the next session

Written 2026-08-04 overnight, from what the first paid runs actually measured. Total spent
so far: **~$1.40**, across three short instances, all destroyed and verified.

## Where we actually are

**Ready and verified locally:**

| | evidence |
|---|---|
| 1B config | preflight clean: 1013M params, 1.03x Chinchilla, decode verified |
| corpus | 21,000,001,090 tokens in B2, manifests re-verified against remote sizes |
| DDP | ws=1 and ws=2 produce identical loss curves AND identical parameters, for AdamW **and Muon** |
| preemption survival | uninterrupted run and run-killed-at-step-30 both end at val_loss 7.591 |
| supervisor | hard spend cap (persisted), verified destroy, two-phase stall detection |
| tests | 917 passing |

**Measured on rented hardware:**

| | value |
|---|---|
| 1B @ 32k, 1x H100 SXM | 17,288 tok/s, 28.3% MFU at 8ND |
| `grad_checkpoint=False` | **OOMs even on 80 GB** — 8ND is not optional |
| B2 -> instance, US-West | 69 MB/s |
| B2 -> instance, Asia | 12.2 MB/s |
| bid vs on-demand | ~0.30x, but host-specific and volatile (4x H100: $1.81 -> $3.56/h in hours) |

**The one thing we still do not know: multi-GPU scaling efficiency.** Every cost estimate
for a 4x run assumes ~96%, unmeasured. That is the single highest-value measurement
outstanding and it costs about $3.

## The decision that shapes everything else

**Do the cheap engineering BEFORE the capstone, not after.** With today's tooling a
preemption costs 23.5 minutes of dead time in US-West. Over a 3.5-day run at an assumed
6-hour MTBF that is ~14 preemptions = 5.5 hours = **~16% overhead**, and it forces us onto
expensive US hosts.

Lazy shard fetch plus a pre-built image takes that to ~3 minutes: **~2% overhead**, and it
re-opens the cheap non-US hosts. That is roughly half a day of free local work against
~$100-200 of avoided waste and a materially calmer run.

## Morning sequence

### Step 1 — the scaling measurement (~25 min, ~$3)

```
python scripts/vast_supervisor.py --gpus 4 --max-price 1.75 --max-spend 20 \
  --target-step 60 --run-prefix smoke-4x --config configs/coder-1b-smoke4x.py \
  --min-reliability 0.95 --min-disk 200 --host-id 456754 --geo US \
  --setup-grace-minutes 45 --stall-minutes 20 --poll 90 --yes
```

Pinned to the Washington host (69 MB/s to B2, 0.998 reliability) because for a 25-minute
test the corpus pull dominates. Both bugs that killed the previous attempts are fixed:
the watchdog no longer counts setup as a stall, and `--host-id` stops cheapest-first
selection from taking Asia.

**What it answers:** NCCL across 4 ranks, and tokens/sec versus the 17,288 single-GPU
baseline. Expect ~66,000 tok/s if scaling is ~96%.

**Gate:**
- **>= 85% scaling** (>= 58,800 tok/s) -> 4x is the right shape, proceed
- **70-85%** -> usable; re-price 2x, which has less all-reduce and identical GPU-hours
- **< 70%** -> something is wrong; do not scale up until it is understood

### Step 2 — the engineering that makes bid pricing cheap (half a day, free)

Ordered by value per hour of work:

1. **Lazy shard fetch with local cache.** `ShardDataset` already addresses shards by index
   and training reads them in random order, so nothing needs all 105 present to start.
   Time-to-first-step: 9 min -> ~6 s in the US, 54 min -> ~6 s in Asia. This is the item
   that re-opens cheap hosts, which matters more than the speedup.
2. **Pre-built Docker image** pinning torch/triton/fla/liger. Removes the remaining ~8 min
   of dependency install, and removes a whole class of failure — the first shakedown died
   because the stock image shipped Triton 3.1 against fla's 3.6 requirement.
3. **Bid at 1.25x the floor, and log preemptions per bid multiple.** We currently bid
   1.02x, which is where anyone outbids us. Two runs turn MTBF from a guess into a number.
4. **Rank offers on bid + expected restart cost + reliability,** not bid alone. This is what
   picked Thailand at $0.89/GPU-h over Washington at $1.73 and then paid 54 minutes for it.
5. **Accept a ranked GPU list.** A100 PCIE 4x sits at $0.13-0.14/GPU-h — a third of H100
   per GPU-hour for ~0.63x the FLOPs, so cheaper per token when H100 bids spike.

### Step 3 — the capstone (~3.5 days, ~$150-300)

Only after 1 and 2. At that point a preemption costs ~3 minutes and host choice is free,
so bid pricing is genuinely cheap rather than nominally cheap.

```
python scripts/vast_supervisor.py --gpus 4 --max-price 0.60 --max-spend 350 \
  --target-step 40000 --run-prefix coder-1b --config configs/coder-1b.py \
  --setup-grace-minutes 45 --stall-minutes 30 --yes
```

**Before pressing go, write down the expected loss curve** at 2B/5B/10B/21B tokens, from
the frontier-32k run and the original 1B. Without it "is this working?" stays a judgement
call for three days; with it, hour two is falsifiable.

Run `scripts/eval_suite.py` against each milestone as it lands — per-slice val loss, FIM
middle-span loss, syntax validity, repetition, and the probe battery including the code and
math categories.

## Two cheap experiments worth slotting in

Both can ride along on a box we have already rented:

- **Uncheckpointed activation size at 32k.** We know it exceeds 61.8 GB (that is what
  OOM'd) but not by how much. If the true figure is ~65 GB then 8-bit optimizer states
  (freeing 9.1 GB) would let us drop checkpointing entirely: 8ND -> 6ND, ~25% fewer FLOPs.
  If it is 85 GB, nothing helps. One measurement settles it.
- **8-bit Muon momentum A/B.** 8-bit Adam is well validated; 8-bit *Muon* is not, and Muon
  orthogonalises via an iterative Newton-Schulz on the momentum buffer, which is plausibly
  more precision-sensitive. This matters far beyond the 1B: 8-bit states are what would take
  a 6B from "needs FSDP and multi-node" to "fits one card". It needs a real A/B, because a
  quantisation that quietly degrades the optimizer produces a perfectly plausible loss
  curve.

## Standing rules for paid runs

- **Never leave a box without a supervisor.** Its `finally` is the teardown; if it dies, the
  box bills on. Confirm with `vast_run.py instances`, and confirm against the API rather
  than our own tooling when it matters.
- **Verify destroy, do not trust the response.** Vast's DELETE has already returned success
  while the instance kept running. `destroy()` now re-checks the instance list.
- **Measure rates, do not extrapolate from one reading.** "3.5 GB at 14 minutes" told us
  nothing; two samples 60 s apart gave 12.2 MB/s and changed the decision.
- **Report elapsed time and dollars in every status update.**

## Longer horizon, from docs/scaling-3b-6b.md

3B is the reachable next step at ~$1,200-2,400 and needs ~11B more tokens than we hold. 6B
needs the corpus roughly tripled and either 8-bit states or FSDP, at $5,700-22,000. Cost
grows with the **square** of parameters, and **data binds before money does**.
