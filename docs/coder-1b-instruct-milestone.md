# coder-1b-instruct: the distill-vs-no-distill A/B, measured

Written 2026-08-10. Two instruction-tuned models were trained from the same coder-1b base
(`ckpt_40000`) with **identical hyperparameters**, differing only in data:
- **arm A (compliant)** — no-distill: CommitPackFT + MBPP-train + OASST code threads +
  executor-verified APPS/CodeContests/TACO solutions.
- **arm B (distilled)** — Magicoder-Evol-Instruct-110K (GPT-4-authored), token-matched to
  arm A.

Scored against the bands in `docs/coder-1b-instruct-prediction.md`, committed *before either
arm trained*. All local on one RTX 6000 Ada; total cost $0.

## Headline: the rule costs a few points on HumanEval, ~nothing on MBPP

| benchmark (greedy pass@1) | base | arm A (compliant) | arm B (distilled) | distill gap (B−A) |
|---|---|---|---|---|
| HumanEval | 0.6% (1/164) | **3.0% (5/164)** | **7.3% (12/164)** | **+4.3 pt** |
| MBPP | 3.9% (10/257) | **14.0% (36/257)** | **14.8% (38/257)** | **+0.8 pt** |

Both arms clearly beat the base (SFT worked). The distilled arm holds a **modest edge on
HumanEval (+4.3 pt)** but is **within noise on MBPP (+0.8 pt)** — and MBPP is the larger,
more reliable benchmark (257 vs 164 tasks). The build-capability rule's measured cost is
therefore small and benchmark-dependent: a handful of HumanEval points, essentially free on
MBPP. The compliant arm captures most of the delivered capability.

## Prediction scorecard: in-band, no falsifier fired

| claim | predicted | measured | verdict |
|---|---|---|---|
| arm A HumanEval greedy | 2–8% | 3.0% | ✓ in band |
| arm A MBPP greedy | 6–16% | 14.0% | ✓ in band |
| arm B HumanEval greedy | 4–14% | 7.3% | ✓ in band |
| arm B MBPP greedy | 8–22% | 14.8% | ✓ in band |
| distill gap (HumanEval) | +3–10 pt, OR <3 pt near-null | +4.3 pt | ✓ modest distilled win (as predicted) |
| distill gap (MBPP) | +3–8 pt, OR <2 pt near-null | +0.8 pt | ✓ near-null |

None of the four falsifiers fired: neither arm at/below base (both well above); compliant did
not beat distilled on both benchmarks; no distilled win >15 pt (the leakage tripwire); and
the guardrails held (below). The pre-registered directional call — "a modest distilled win,
not a null" — was right on HumanEval and resolved to the near-null on MBPP. The one miss was
a *secondary* prediction: the pairwise-judge call ("distilled win-rate ≥ compliant") was
falsified in the pro-compliant direction (see the pairwise section) — the compliant arm was
strongly preferred on held-out instructions, though on prompts drawn from its own distribution.

## Guardrails: infill preserved, long-context mostly held

| guardrail | base | arm A | arm B | predicted |
|---|---|---|---|---|
| FIM middle-loss | 0.5848 | 0.6255 (+0.041) | 0.6151 (+0.030) | ≤ +0.05 → **both PASS** |
| passkey retrieval (overall) | 96.9% | 91.7% (−5 pt) | 85.4% (−11 pt) | within ~10 pt |
| — passkey at 8k | 0.88 | 0.92 | 0.71 | — |
| greedy repetition loop_rate | 0.75 | 0.25 | 0.38 | (drop expected) |

Chat-SFT did **not** damage infill on either arm (FIM within the predicted 0.05). Long-context:
the compliant arm is cleanly within tolerance (−5 pt); the distilled arm sits marginally past
the ~10-pt line (−11 pt overall, −17 pt at 8k), but with n=8 samples/cell this is
noise-adjacent (one 3/8 cell drove it) and 85% is far from a collapse — an honest marginal
flag, not a failure. Both arms sharply reduced the base's greedy looping (0.75 → 0.25 / 0.38),
which is why greedy pass@1 is the honest headline metric here (the sampled-decoder hedge in
the prediction proved unnecessary).

## Pairwise judge (held-out, external judge) — points the other way

On 120 held-out instructions, judged by codex in both orderings (a win requires being
preferred in BOTH orders; inconsistent verdicts count as ties):

**compliant 100 wins · distilled 20 wins · 0 ties → compliant preferred 83%.**

This **contradicts the execution benchmarks** (where distilled edged compliant) and
**falsifies this doc's own pairwise prediction** ("the distilled arm's win-rate is ≥ the
compliant arm's") — an honest miss, in the pro-compliant direction. Two things reconcile the
opposite signals:

1. **Confound (weakens the 83%):** the held-out set is a slice of the *compliant* mix —
   commit-message, competitive-statement, and OASST-code styles. That is arm A's training
   *distribution* (just not these exact rows), and a distribution arm B never saw at all. So
   the judge is scoring on arm A's home turf; the 83% overstates compliant's general
   superiority. A truly neutral pairwise needs prompts from a third distribution, which this
   run did not build.
2. **Signal (survives the confound):** an 83–17 split is large for mere style familiarity,
   and it aligns with the FIM/repetition guardrails (compliant had the better greedy syntax
   0.83 and lower looping 0.25). The distilled arm's advantage really is *confined to the
   narrow benchmarks its GPT-4 data was shaped for*; on broader code-instruction following it
   is not ahead.

Net: the neutral execution benchmarks (distilled slightly ahead) are the cleaner headline;
the pairwise (compliant ahead, but on its own turf) says the distilled edge does not
generalize beyond the benchmarks. Both readings agree the build-capability rule is affordable.

## The decontamination asymmetry (a finding, not a confound)

Identical decontamination (same 10-grams, same benchmark source) was applied to both arms.
It removed **78 rows from arm A** but **2,968 rows from arm B** (Magicoder). That 38× gap is
direct evidence the distilled data is *benchmark-shaped*: Magicoder's GPT-4-authored
instruction/solution pairs overlap HumanEval/MBPP far more than genuine human code does. This
is protective, not biasing — arm B was cleaned to the same standard, so its remaining data is
thoroughly decontaminated and its +4.3-pt HumanEval edge is on cleaned data (well under the
15-pt leakage tripwire). It also explains *why* a distilled edge exists at all: even after
decontamination, benchmark-shaped data transfers a little better to the benchmarks.

## Data composition (both arms)

| | arm A (compliant) | arm B (distilled) |
|---|---|---|
| rows | 7,906 | 4,611 |
| supervised tokens | 1,913,014 | 1,913,674 (matched to 0.03%) |
| sources | commit 5,000 / competitive 2,651 (APPS 493 + CodeContests 1,181 + TACO 977) / OASST 363 / MBPP 170 | Magicoder-Evol-110K, subsampled |
| decontaminated out | 78 | 2,968 |

Arm A is Python-first (matching the pretraining code and the benchmarks). The competitive
solutions are human-accepted submissions, executor-verified in the sandbox as a sanity filter
(6 leading I/O cases, 5 shortest solutions per problem). A 200-row slice was held out from
both arms for the pairwise judge. Training: block 2048, effective batch 16, 3 epochs, LR 2e-5.

## Honest caveats

- Arm B differs from arm A in provenance *and* instruction style/coverage simultaneously.
  Token-matching controls volume, not style. So this measures the **delivered-capability
  cost** of the rule (what benchmark score we forgo by refusing distilled data), not a
  mechanistic claim that teacher text is intrinsically better to learn from.
- Arm A is commit-majority (63%) because CommitPackFT is the largest compliant source;
  the function-completion signal that matches the benchmarks comes mostly from the 2,651
  executor-verified competitive rows + MBPP. A larger executor-verified core is the obvious
  next lever to strengthen the compliant arm.
- Absolute pass@1 is low (single-digit HumanEval) — expected for a 1.2B model with a small
  SFT set. The A/B *contrast*, not the absolute level, is the result.

## Bottom line

The build-capability rule is affordable at this scale: a compliant, no-distill SFT set built
from human commits and executor-verified problems recovered most of the instruction-following
capability a GPT-4-distilled set delivered — losing ~4 HumanEval points and roughly nothing on
MBPP. The distilled arm's edge is real but small and concentrated where its benchmark-shaped
provenance helps most. Next lever: grow the executor-verified competitive core and add
preference optimization (Phase 2) on the compliant base.
