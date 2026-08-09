# Written BEFORE training: what coder-1b-instruct's A/B should show

Committed 2026-08-09, **before either SFT arm is trained**, following the same
pre-registration discipline as `docs/coder-1b-prediction.md`. The point is to make the
distill-vs-no-distill question falsifiable in advance, so the result is a measurement and
not a story told after the numbers land.

The experiment (design: `docs/superpowers/specs/2026-08-09-coder-1b-instruct-design.md`):
instruction-tune the coder-1b base (`ckpt_40000`) on two data mixes from the *same base with
identical hyperparameters* — **arm A** (compliant: CommitPackFT + MBPP-train + OASST code
threads + executor-verified APPS/CodeContests/TACO) and **arm B** (distilled:
Magicoder-Evol-Instruct-110K, GPT-4-authored, token-matched to arm A). Only the data differs.

## The base floor (measured, `ckpt_40000`, greedy)

| metric | base greedy | note |
|---|---|---|
| HumanEval pass@1 | **0.61% (1/164)** | near the floor for a 1.2B base, no instruction tuning |
| MBPP pass@1 | **3.89% (10/257)** | run-best; jitters task-to-task at this scale |
| FIM middle-loss | **0.5848** | the base's real strength; the guardrail metric |

The base's sampled HumanEval/MBPP floor is measured as part of the Task 11 battery (same
`eval_code.py --mode chat` path both arms use); this prediction states bands for both
decoders regardless.

## The prediction

**Framing.** SFT on a code base lifts execution benchmarks — but this is a 1.2B model with
a modest, single-epoch-scale compliant set, so gains are real yet bounded, not the 2–5×
jumps seen at 7–15B. Greedy stays below sampled on this model class (greedy argmax loops —
the run-long [[repetition-is-decoder-side]] finding), so sampled is the honest number.
Magicoder is *benchmark-shaped* (its GPT-4 authorship optimized instruction/solution pairs
that resemble HumanEval/MBPP), which is exactly why the distilled arm is expected to edge
the compliant one on these specific benchmarks.

| metric | arm A (compliant) | arm B (distilled) |
|---|---|---|
| HumanEval pass@1, greedy | 2 – 8% | 4 – 14% |
| HumanEval pass@1, sampled | 4 – 12% | 6 – 18% |
| MBPP pass@1, greedy | 6 – 16% | 8 – 22% |
| MBPP pass@1, sampled | 8 – 20% | 10 – 26% |

**Distill-gap (the headline).** The distilled arm beats the compliant arm by **≈3–10
HumanEval points and ≈2–8 MBPP points (sampled)**, OR the gap falls **within ±2 tasks of
noise** (the interesting null). Both outcomes are informative: a clear distilled win prices
the build-capability rule; a null vindicates it. The null is a live possibility — StarCoder2-
Instruct found own-distribution data beat GPT-4-distilled data at 15B — but at 1.2B, where
the base is too weak to benefit much from its own distribution, the benchmark-shaped
distilled data is expected to win modestly. We predict a modest distilled win, not a null.

**Guardrail.** Both instruct arms keep FIM middle-loss within **~0.05 of the base (≤ ~0.635)**
— chat-SFT at block 2048 on a NoPE model should not wreck infill — and neither collapses on
a passkey long-context probe.

**Pairwise judge.** On held-out code instructions, the distilled arm's win-rate is ≥ the
compliant arm's, but by less than the execution-benchmark gap (Magicoder's edge is sharpest
on benchmark-like tasks; on general code Q&A the human-authored compliant data is more
competitive).

## What would falsify it

- **Either arm at or below the base** (HumanEval ≤ 0.61%, MBPP ≤ 3.89%, sampled) — SFT is
  broken (bad masking, wrong LR, corrupted data), not a data-provenance result.
- **A compliant arm that beats the distilled arm by more than noise on both benchmarks** —
  contradicts the benchmark-shaped-data expectation; would be a strong (and publishable-
  feeling) result for the build-capability rule at this scale.
- **FIM middle-loss materially above the base (> ~0.65) on either arm** — chat-SFT damaged
  the base's core strength; block size too small or catastrophic forgetting, revisit before
  reading the A/B.
- **Decontamination removes wildly different fractions from the two arms** — the comparison
  is confounded; a distilled "win" on a less-decontaminated arm is contamination, not
  capability. The builders' per-source removal reports must be comparable.

## Caveat stated up front

Arm B differs from arm A in provenance *and* instruction style/coverage simultaneously.
Token-matching controls for volume, not style. So this measures the **delivered-capability
cost** of the build-capability rule (what we give up in benchmark score by refusing distilled
data), not the mechanistic claim that teacher-authored text is intrinsically better to learn
from. That practical question is the one worth pricing.
