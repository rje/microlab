# What a 3B or 6B would cost, from measured numbers

Written 2026-08-04. Every figure below is derived from one measurement — **17,288 tok/s for
the 1B at 32k on a single H100 SXM, which is 28.3% MFU at 8ND** — rather than from vendor
peak FLOPs. The 1B's own predicted cost reproduces what we observed (326 vs 337 GPU-h
measured), so the extrapolation has at least one anchor.

8ND rather than the usual 6ND because **gradient checkpointing is mandatory at 32k** — it
OOMs without it even on an 80 GB H100 (measured, not assumed).

## Shapes and what they fit on

Scaling the architecture at head_dim 128, keeping the 3:1 KDA:MLA hybrid and NoPE:

| | layers | d_model | heads | params | optimizer states | activations @32k | total | runs on |
|---|---|---|---|---|---|---|---|---|
| **1B** | 24 | 1792 | 14 | 1.01B | 18.2 GB | 9.0 GB | **27.2 GB** | 1x H100 80GB |
| **3B** | 32 | 2688 | 21 | 2.91B | 52.3 GB | 18.0 GB | **70.3 GB** | 1x H100, *tight* |
| **6B** | 40 | 3584 | 28 | 6.34B | 114.2 GB | 30.0 GB | **144.2 GB** | **FSDP required** |

States are 18 bytes/param (bf16 2 + fp32 master 4 + AdamW m,v 8 + Muon momentum 4).
Activations use a 3.19x constant on `L * T * d * 2`, calibrated against the 1B's measured
27.2 GB peak.

### The 6B finding that changes the plan

**6B's optimizer state alone is 114 GB. It does not fit on an 80 GB H100 before a single
activation is stored.** DDP replicates optimizer state on every rank, so DDP — which is what
we just built and validated — **cannot train a 6B on H100s at all**. Options:

- **FSDP / ZeRO-2+**, sharding optimizer state across ranks. New work, and a materially
  larger surface than DDP: FSDP changes checkpointing, resume, and the world-size-invariance
  property the migration design depends on.
- **H200 (141 GB)** — still short once activations are counted, so FSDP anyway.
- **B200 (179-192 GB)** — fits, at ~$5-6.70/h per GPU observed.
- **8-bit optimizer states**, which would take 114 GB to ~66 GB and put 6B back inside a
  single H100. Cheapest path by far, and worth evaluating before committing to FSDP.

3B at 70.3 GB fits an 80 GB card but with under 10 GB of headroom — enough to train, not
enough to be relaxed about. An H200 or 8-bit states would make it comfortable.

## Compute and money

Chinchilla-optimal `D = 20N`. Costs at the three bid prices we have actually seen for
H100 SXM: $0.45/GPU-h (Japan 1x), $0.89 (Thailand 4x), $1.73 (Washington 4x, no discount).

| | training FLOPs | GPU-hours | @ $0.45 | @ $0.89 | @ $1.73 | 4x H100 wall-clock |
|---|---|---|---|---|---|---|
| **1B** | 1.64e20 | 326 | **$147** | $290 | $564 | 3.5 days |
| **3B** | 1.35e21 | 2,683 | **$1,207** | $2,388 | $4,641 | **29 days** |
| **6B** | 6.44e21 | 12,770 | **$5,746** | $11,365 | $22,091 | **138 days** |

Scaling is brutal and worth stating plainly: **3B costs ~8x the 1B, and 6B ~39x.** That is
`N * D = 20N^2`, so cost grows with the *square* of parameters at Chinchilla-optimal
training.

Wall-clock on 4 GPUs makes 3B and 6B impractical at that width. Realistic configurations:

| | GPUs | wall-clock | why |
|---|---|---|---|
| 3B | 8x H100 | ~15 days | needs FSDP or 8-bit states for headroom |
| 3B | 16x H100 | ~7 days | interconnect starts to matter |
| 6B | 32x H100 | ~17 days | FSDP mandatory; multi-node |
| 6B | 8x B200 | ~14 days | fits without sharding, ~$6/GPU-h |

Multi-node is a step change in complexity we have not touched: NCCL over the network,
node-failure handling, and a preemption model where losing one node kills the job.

## Data is the binding constraint, before money

We hold **47.6B tokens** across all six slices (code 27.35B, web 8.0B, arXiv 7.4B, math 3.0B,
markdown 1.5B, commits 0.33B).

| | needs (20N) | vs what we have |
|---|---|---|
| 1B | 20.3B | **OK** |
| 3B | 58.1B | short by 11B — **1.2x** more needed |
| 6B | 126.8B | short by 79B — **2.7x** more needed |

3B is nearly reachable: another ~11B tokens of web or code closes it, and FineWeb alone can
supply that. 6B needs the corpus roughly tripled, which is a real data-engineering project —
and at that point the *mix ratios* matter more than the totals, because the code slice
cannot grow 2.7x from the-stack permissive subset without either repeating or loosening the
licence filter.

**Repeating data is not free.** Muennighoff et al. put the useful limit at ~4 epochs before
returns fall off sharply; our own repetition lane exists to measure where that bites for us.
Training a 6B on 47.6B tokens repeated 2.7x is a legitimate option, but it is a different
experiment from a Chinchilla-optimal 6B and should be labelled as such.

## What I would actually recommend

**3B is the reachable next step; 6B is not, at current corpus and tooling.**

Sequence, if the 1B lands well:

1. **Finish the 1B.** ~$150-300 and 3.5 days on 4x H100. It is the only one of the three
   whose data budget we already meet.
2. **Evaluate 8-bit optimizer states on the 1B.** Cheap to test, and it is what makes 3B
   comfortable and 6B single-card-feasible. This is the highest-leverage piece of
   engineering for scaling, ahead of FSDP.
3. **Grow the corpus to ~60B** — mostly FineWeb, plus whatever the-stack can still yield
   under the licence filter. Needed for 3B regardless of when we run it.
4. **Then 3B at ~$1,200-2,400**, 8-16x H100, ~1-2 weeks.

6B should stay a written plan until (a) the corpus reaches ~130B tokens, (b) either 8-bit
states or FSDP is proven, and (c) there is a reason to prefer it over a better-trained 3B.
At $5,700 on the optimistic bid price and $22,000 on the pessimistic one, the variance in
the *rental market* alone exceeds the entire cost of the 1B run.

## Caveats on these numbers

- **28.3% MFU is our number, at 1B.** Larger models usually achieve *higher* MFU (better
  arithmetic intensity), so these GPU-hours are likely conservative — possibly by 20-30%.
- **Multi-GPU scaling above 4 ranks is unmeasured.** The 4x figure already assumes ~96%
  efficiency we have not confirmed; at 16-32 ranks, with the 2.03 GB/step all-reduce
  growing in proportion to parameters, that assumption gets weaker.
- **Bid prices are volatile.** A 4x H100 moved $1.81 -> $3.56/h within one session. Treat
  the $0.45 column as best-case-if-you-wait and the $1.73 column as what you pay if you
  need capacity now.
