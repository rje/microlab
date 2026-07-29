# Peri-LN verdict + the lab's cross-seed noise band

Four 124M arms, 4500 steps each, FineWeb-100BT, identical except `block_norm` and `seed`:
`runs/periln-ab-{pre,peri}` (seed 1337) and `runs/periln-ab-{pre,peri}-s1338` (seed 1338).
Analysis: `scripts/analyze_periln_ab.py`.

## Results

Final val loss at step 4500:

| | seed 1337 | seed 1338 | seed effect |
|---|---|---|---|
| **Pre-LN** | 3.2878 | 3.3026 | +0.0148 |
| **Peri-LN** | **3.2755** | **3.2859** | +0.0104 |
| **intervention (peri − pre)** | **−0.0123** | **−0.0167** | |

Perplexity: 26.78 / 27.18 (pre), 26.46 / 26.73 (peri).

## Verdict: ADOPT Peri-LN, on reproducibility of direction rather than size of effect

Peri-LN wins at **both seeds and all 36 matched eval points** (18 per seed). Mean intervention
effect **−0.0145 nats**. The advantage is largest early (−0.069 at step 250) and **decays
monotonically** to −0.012–0.017 by 4500 — Peri-LN mostly buys faster early convergence, and the
curves are converging. Do not extrapolate this as a fixed advantage at longer training.

## The more important result: our noise band was wrong by 10x

**Cross-seed spread is the same size as the intervention effect.** Changing only the seed moves
final val loss by 0.0148 (pre) / 0.0104 (peri); changing the architecture moves it by 0.0123 /
0.0167. A single-arm-per-config comparison at this scale **cannot distinguish an architecture win
from a lucky seed.** Peri-LN survives only because the sign is consistent across both seeds and
every eval point.

This corrects a number the lab had been quoting. We previously used a **0.0014 nats** band from
config-identical twin runs (`muon-ab-muon` vs `nope-ab-rope`). That measures *kernel
nondeterminism at a fixed seed* — not init and data-order variation — and it is the wrong
denominator for comparing two architectures, because a different architecture is effectively a
fresh random draw.

| band | what it measures | value |
|---|---|---|
| twin-run | kernel nondeterminism, same seed | 0.0014 (max), 0.0009 (mean) |
| **cross-seed** | **init + data order — use this for arch comparisons** | **~0.013** |

Consequence, already applied to `nope-verdict-audit.md`: the NoPE-vs-RoPE gap of +0.057 is
**~4.4x** the cross-seed band, not the ~40x originally claimed. That verdict still stands, but the
honest multiplier is an order of magnitude smaller.

**Rule going forward: any "Nx the noise floor" claim uses the cross-seed band (~0.013 at
124M/4500 steps). Any architecture A/B at this scale needs ≥2 seeds per arm, or its verdict is
PROVISIONAL by construction.**

## What this experiment did NOT test

We adopted Peri-LN primarily for its **variance-reduction** claim — the literature reports it cuts
seed-to-seed benchmark std by more than half, which would de-noise every ablation we run. Our
numbers are directionally consistent (peri's cross-seed spread is 0.0104 vs pre's 0.0148, ~70%),
but that is **one difference of differences from n=2 seeds per arm** — nowhere near enough to
confirm, and the published claim was about downstream benchmark std over 5 seeds, not val loss over
2. Treat the variance benefit as **unverified**; a third seed per arm would make it a real
measurement.
