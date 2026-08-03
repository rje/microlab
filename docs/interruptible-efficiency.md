# Getting the most out of interruptible machines

Written 2026-08-04 from measurements taken during the first paid runs, not from theory.
Bid pricing is ~30% of on-demand, which is the difference between a $600 run and a $160
one. Everything here is about making that discount safe to take.

## What we measured

| quantity | value | where |
|---|---|---|
| 1B @ 32k, 1x H100 SXM | 17,288 tok/s, 28.3% MFU (8ND) | shakedown |
| B2 -> instance, US-West | **69 MB/s** | Washington host |
| B2 -> instance, Asia | **12.2 MB/s** | Thailand host |
| B2 upload (home) | 32.5 MB/s | corpus push |
| checkpoint | 16.6 GB | 1B, params+optimizer |
| bid vs on-demand | 0.30x typical, but **host-specific** | 4x H100 seen at $1.79 bid / $8.53 on-demand |

## 1. Restart cost is the thing to attack, not checkpoint frequency

A preemption costs two separate things: the *work since the last checkpoint*, and the
*fixed cost of getting a new box productive*. The second is bigger and is the one nobody
budgets for.

| region | deps | corpus pull | ckpt download | total | 4x GPU dead time |
|---|---|---|---|---|---|
| US-West | 8 min | 9 min | 4.0 min | **23.5 min** | 1.57 GPU-h = $0.70 |
| Asia | 8 min | 54 min | 22.7 min | **86.2 min** | 5.75 GPU-h = $2.59 |

**The single highest-value change: fetch shards lazily instead of pulling the whole
corpus.** The corpus is 105 shards of ~400 MB, and training reads them in RANDOM order —
`sequence_at()` picks a shard per sequence. Nothing needs all 105 present to start.

- pull-everything: **9 min** before step 1 (US-West), 54 min from Asia
- fetch-on-demand + local cache: **~6 s** before step 1

That takes US-West restart cost from 23.5 min to ~11 min, and Asia from 86 min to ~11 min —
which also **removes the geographic constraint on host choice**, worth more than the
speedup because it re-opens the cheap non-US hosts for the capstone.

Implementation: `ShardDataset` already addresses shards by index. Wrap the memmap open in a
"present locally? else fetch from B2, then memmap" step, with a background prefetch of the
next few shards. The manifest already carries every shard's size, so integrity checking is
unchanged.

**Second-order: cache the pip environment.** 8 minutes of every restart is dependency
installation, including a torch upgrade. Building one image with the right torch/triton/fla
and pushing it to a registry removes that entirely. Vast pulls images fast.

## 2. Checkpoint cadence is already about right

Young/Daly gives the optimal interval as `tau = sqrt(2 * delta * MTBF)`, with `delta` the
cost of a checkpoint (4.6 min for 16.6 GB at 60 MB/s):

| MTBF | tau_opt | checkpoints/day | avg work lost per preemption |
|---|---|---|---|
| 2 h | 33 min | 43 | 17 min |
| 6 h | 58 min | 25 | 29 min |
| 12 h | 81 min | 18 | 41 min |
| 24 h | 115 min | 13 | 58 min |
| 48 h | 163 min | 9 | 81 min |

Our 250-step cadence is **33 min at 4x H100** — optimal for a 2-hour MTBF and conservative
beyond that. No change needed until we know the real preemption rate, which we should
measure rather than guess: the supervisor already records every episode in
`runs/.supervisor-state.json`, so MTBF falls out of the first long run for free.

**Do not shorten it.** Because uploads are async, a shorter interval does not cost training
throughput, but it does multiply B2 writes and — more importantly — narrows the window in
which a half-written checkpoint can be picked up. The syncer's size-stability check handles
that, and there is no reason to lean on it harder.

## 3. Bid strategy: the floor is not the right bid

`min_bid` is the price at which you are outbid by literally anyone. Bidding modestly above
it buys MTBF, and MTBF is worth real money through the restart cost above.

At 4x H100, a restart costs ~$0.70 (US, with lazy fetch: ~$0.35) plus lost work. If bidding
20% over the floor doubles MTBF, on a 3.5-day run it pays for itself as long as it prevents
roughly one preemption per day. The supervisor currently bids `min_bid * 1.02`, which is
close to the floor and near-maximally preemptible.

**Recommended: bid `min_bid * 1.25`, and record preemptions against the multiple.** Two or
three runs give an empirical MTBF-vs-bid curve, which is the only honest way to price this.

## 4. Diversify across hosts, and re-price every provision

The market moved three times in one session: a 4x H100 in Thailand went $1.81 -> $3.56/h in
a few hours; Montana and Washington 1x offers appeared and vanished. The supervisor already
re-searches before every provision, which is what let it survive that. Two additions worth
making:

- **Rank on `bid + expected_restart_cost(region)`,** not bid alone. Today's selection took
  Thailand at $0.89/GPU-h over Washington at $1.73 and paid for it in a 54-minute corpus
  pull. With lazy fetch this mostly evaporates, which is the point.
- **Fall back across GPU types.** A100 PCIE 4x sits at $0.13-0.14/GPU-h — roughly a third
  of H100 per GPU-hour for ~0.63x the FLOPs, so *cheaper per token* if available. The
  supervisor should accept a ranked list of acceptable GPUs, not one.

## 5. What NOT to do

- **Network volumes.** Region-pinned and priced per GB-month; they solve the corpus problem
  worse than lazy fetch does, and they constrain host choice in exactly the way we are
  trying to escape.
- **Shrinking checkpoints by dropping optimizer state.** Resuming without it is not the
  same run. If checkpoint size becomes the bottleneck, 8-bit optimizer states are the
  legitimate lever (halves the 12N of AdamW+Muon), not omission.
- **Chasing the absolute cheapest host.** Reliability below ~0.98 on a multi-day run means
  more restarts, and each one costs real money; the headline rate is not the price.

## 6. Concrete backlog, ordered by value

1. **Lazy shard fetch with local cache** — removes 9-54 min from every restart and re-opens
   non-US hosts. Biggest single win.
2. **Pre-built Docker image** with torch/triton/fla/liger pinned — removes another 8 min.
3. **Bid at 1.25x floor, log preemptions** — turns MTBF from a guess into a measurement.
4. **Rank offers on total expected cost**, including restart cost and reliability.
5. **Accept a ranked GPU list** so A100 capacity is usable when H100 bids spike.

Items 1-2 together take a US restart from 23.5 min to ~3 min, and an Asian one from 86 min
to ~3 min. That is what makes bid pricing genuinely cheap rather than nominally cheap.
