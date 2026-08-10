# Written BEFORE training: what the Phase 2 free levers should buy

Committed 2026-08-10, **before any Phase 2 training stage runs**, per the lab's
pre-registration discipline (`docs/coder-1b-prediction.md`, `docs/coder-1b-instruct-
prediction.md`). The program (spec: `docs/superpowers/specs/2026-08-10-coder-1b-phase2-
design.md`): executor-reward GRPO + correctness-contrast IPO from v1 independently, stack
per rule, then self-gen — all local, $0.

## The v1 baseline (measured)

| metric | v1 (`runs/coder-1b-instruct-compliant`) |
|---|---|
| HumanEval greedy pass@1 | 3.0% (5/164) |
| HumanEval sampled pass@1 / pass@10 (t=0.7, k=40) | 1.2% / **6.1%** |
| MBPP greedy pass@1 | 14.0% (36/257) |
| FIM middle-loss | 0.6255 |
| passkey overall (n=8/cell) | 91.7% |
| error modes | wrong-logic 61–67% of failures; format 3–9% |

Distilled reference (the Phase-1 arm B): HumanEval 7.3% / MBPP 14.8% greedy. **Program
success = a compliant model beating it on both.**

## Predictions

**Executor-GRPO** (from v1): the mechanism is selection — RL reweights the policy toward
what passes, bounded above by roughly the policy's pass@k. Bands:
- MBPP greedy: **18–22%** (from 14.0)
- HumanEval sampled pass@1: **4–5%** (from 1.2; ceiling the 6.1% pass@10)
- HumanEval greedy: **4–9%** (from 3.0; greedy jitter ±1–2 tasks applies)
- Training telemetry: mean group reward RISES over iterations; distinct-fraction does not
  collapse (mode-collapse watch).

**Contrast-IPO** (from v1, independent): **+1–3 MBPP points** over v1; HumanEval within
jitter of v1. Smaller than GRPO — it shifts the policy off near-misses but adds no
selection pressure.

**Stack** (only if both beat v1; smaller-gain lever re-run from the winner): **≥ the best
solo arm**; kept only if it beats the solo winner.

**Self-gen** (from the best checkpoint): **+1–4 MBPP points** over its base checkpoint;
yield prediction: ≥15% of full-pool problems produce a full-pass solution at k=8 (v1's
pre-pass rate, rising with whatever GRPO bought).

**Guardrails, every stage:** FIM middle-loss ≤ **0.68**; passkey overall ≥ **82%**; codex
pairwise vs v1 (held-out, both orderings) ≥ **40%** win-rate for the new model — execution
training must not wreck general instruction-following.

## What would falsify it

- **Any stage below v1 on BOTH benchmarks** — the lever failed; revert to the previous
  checkpoint and record it (a lever that hurts is a result, not a tuning excuse).
- **Reward/benchmark divergence** — train reward rising while MBPP falls at the eval gate =
  reward hacking (hardcoding printed outputs, degenerate rollouts). Stop, inspect
  samples.jsonl top-reward rollouts (the prose run's hacking was caught exactly by
  eyeballing), fix, restart.
- **MBPP > 28% at any stage** — too good; a *warning*, not a win (the leakage tripwire, cf.
  both prior prediction docs). Check the pool/benchmark decontamination before believing it.
- **FIM > 0.68 or passkey < 82%** — the stage damaged the base's core strengths; revisit
  before reading the headline numbers.
- **Codex pairwise vs v1 < 40%** — execution-RL narrowed the model into a benchmark
  specialist; that trade must be surfaced, not silently accepted.

## Caveats stated up front

- GRPO's bands assume **no new capability, only selection** — the honest ceiling is the
  policy's own pass@k. If pass@10 also rises materially under GRPO, that's a bonus beyond
  the mechanism this prediction relies on, and should be called out.
- The signal-bearing pool is v1-relative; as the policy improves mid-training, some
  problems drift out of the mixed-group regime. The bands absorb this (it flattens late
  gains); a starved run (mean reward variance → 0 early) is instead a falsifier-adjacent
  warning to inspect.
- Self-gen trains on the model's own distribution filtered by the executor; the +1–4 band
  assumes the echo risks (quirk amplification) are contained by dedup + the guardrails.
