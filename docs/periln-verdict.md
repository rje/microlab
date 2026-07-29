# Peri-LN verdict + the lab's three noise bands

Six 124M arms, 4500 steps each, FineWeb-100BT, identical except `block_norm` and `seed`:
`runs/periln-ab-{pre,peri}[-s1338][-s1339]` (seeds 1337/1338/1339).
Analysis: `scripts/analyze_periln_ab.py`.

## Results

Final val loss at step 4500:

| | seed 1337 | seed 1338 | seed 1339 | mean | sd |
|---|---|---|---|---|---|
| **Pre-LN** | 3.2878 | 3.3026 | 3.2988 | 3.2964 | 0.0077 |
| **Peri-LN** | **3.2755** | **3.2859** | **3.2822** | **3.2812** | **0.0053** |
| **paired effect (peri − pre)** | −0.0123 | −0.0167 | −0.0166 | **−0.0152** | **0.0025** |

Peri-LN wins at **all three seeds and all 54 matched eval points** (18 per seed).
Paired t = **−10.48** (n=3, df=2; |t| > 4.30 is p < .05) — significant despite n=3, because
the paired differences are tightly clustered.

## Verdict: ADOPT Peri-LN

Mean effect **−0.0152 nats** (ppl 26.64 vs 27.08 at the median seed). The advantage is largest
early (−0.069 at step 250) and **decays monotonically** to −0.012…−0.017 by step 4500 — Peri-LN
buys faster early convergence, and the curves are still converging. Do not extrapolate it as a
fixed advantage at longer training.

## The methodology result: there are THREE bands, and the right one depends on the design

The lab has now quoted three different noise floors for the same ladder. Only one of them is the
correct denominator for a given comparison, and getting this wrong inflated a verdict by ~10x
before it was caught.

| band | what it measures | value (124M / 4500 steps) | use it for |
|---|---|---|---|
| twin-run | kernel nondeterminism, same seed *and* config | 0.0014 max / 0.0009 mean | reproducibility checks only |
| **paired intervention** | **sd of (armB − armA) when both share seed/init/data order** | **0.0025** | **our A/Bs — this is the design we actually run** |
| cross-seed | sd within one config across seeds | 0.0065 pooled (pre 0.0077, peri 0.0053) | comparing runs that do NOT share a seed |

**The paired design is far more powerful than the cross-seed spread suggests.** Cross-seed sd is
0.0077, which is larger than the 0.0152 effect divided by 2 — so an *unpaired* comparison really
could not resolve Peri-LN. But our arms share seed, init, and data order, and the paired
differences (−0.0123, −0.0167, −0.0166) have sd 0.0025. That is why n=3 suffices.

### Correcting two earlier claims in this repo

1. The original **~40x** for the NoPE gap used the twin-run band (0.0014) — kernel
   nondeterminism, the wrong quantity entirely.
2. My first correction to **~4.4x** used the cross-seed *range* from n=2 (0.013). Right kind of
   denominator, but a range from two points is not an sd, and it over-corrected.
3. With n=3 the pooled cross-seed **sd is 0.0065**, making the +0.0573 NoPE gap **8.8x** on the
   conservative (unpaired) denominator. NoPE-vs-RoPE was itself a *paired* comparison, so this is
   a floor, not a ceiling. `nope-verdict-audit.md` now carries 8.8x.

**Rules going forward.** State which band a multiplier uses. For a paired A/B (both arms one
seed) the denominator is the paired-difference sd — estimate 0.0025 until a lane measures its
own. Report the effect at every matched eval point, not just the final one: sign consistency
across 18 points is what made a 0.015-nat effect credible. Two seeds is the minimum for any
verdict; three gives a usable paired t.

## Peri-LN's variance-reduction claim: directionally right, magnitude UNCONFIRMED

This is the property we adopted Peri-LN for — the literature reports it cuts seed-to-seed
benchmark std by **more than half**, which would de-noise every later lane.

Measured: cross-seed sd **0.0053 (peri) vs 0.0077 (pre)** — a ratio of **0.69**, i.e. a 31%
reduction, in the right direction but well short of "more than half."

And it is **not statistically distinguishable from no effect**: an F-test on variances with
(2,2) degrees of freedom needs a ratio beyond ~19 to reject at 95%. Three seeds cannot resolve a
variance claim. Also note the published claim concerns *downstream benchmark* std, not val-loss
std, so this is not even the same quantity.

**Status: adopt Peri-LN on the loss result, which is solid; treat the variance benefit as
suggestive only, and do not use it to justify running fewer seeds on later lanes.**
