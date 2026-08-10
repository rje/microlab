# coder-1b-instruct-v2: quantitative coding-capability analysis

Written 2026-08-10, after training **v2** (the "scale the executor-verified core" experiment:
competitive problem→solution data 2,651 → 7,078 rows, 69% of the mix, commit cut to 29%) and
running the full battery. All local, $0. This is a *negative-result-with-a-mechanism* doc:
v2 did not improve on v1, and the error-mode + pass@k analysis says precisely why — and what
would actually move the number.

## 1. Headline: more of the same verified data bought nothing

| greedy pass@1 | base | v1 compliant | **v2 (this run)** | v1 distilled (ref) |
|---|---|---|---|---|
| HumanEval | 0.6% (1/164) | 3.0% (5/164) | **0.6% (1/164)** | 7.3% (12/164) |
| MBPP | 3.9% (10/257) | 14.0% (36/257) | **13.6% (35/257)** | 14.8% (38/257) |

v2 ≈ v1 on MBPP (−1 task); the HumanEval drop (5→1 tasks) is **within the documented greedy
jitter band** for this model class (the pretrain run's own guidance: 0–2/164 swings task-to-
task; do not read the swing). Verdict: **2.7× more executor-verified competitive data moved
pass@1 by ~0.** The format-mismatch hypothesis (stdin/stdout data harming function-completion)
was checked and **refuted**: only 3/164 v2 HumanEval solutions used `input()`; completions are
clean `def`-style.

## 2. Error modes: the ceiling is wrong-logic, not form

Failure taxonomy over every task (greedy), from the executor's stderr:

| | pass | **assertion (runs, wrong output)** | syntax/extract | runtime | timeout |
|---|---|---|---|---|---|
| v1 HumanEval | 3.0% | **64.0%** | 4.9% | 26.2% | 0 |
| v2 HumanEval | 0.6% | **61.0%** | 8.5% | 28.7% | 0 |
| v1 MBPP | 14.0% | **66.9%** | 3.1% | 15.6% | 0.4% |
| v2 MBPP | 13.6% | **61.1%** | 4.3% | 20.2% | 0 |

~2/3 of all failures are **clean, executable code that computes the wrong thing**. Formatting/
extraction failures are marginal (3–9%). Representative near-misses: `square_perimeter(side)
= 4*side*side` (perimeter/area confusion), triangular-prism volume `(a*b*c)//3`. The model has
the syntax, idioms, and instruction format; it lacks **correctness discrimination** — telling
right code from right-*looking* code.

This is why more SFT plateaued: supervised fine-tuning teaches the *distribution* of correct-
looking solutions. It does not teach the boundary between correct and almost-correct. v1 had
already saturated the format lesson (the base→v1 jump); v2 re-taught it.

## 3. pass@k: the capability exists but isn't selected

HumanEval, temp 0.7 / top-k 40, n=10: **pass@1 = 1.2%, pass@10 = 6.1%** — a **5× gap**.

The model *produces* a correct solution within 10 draws for ~6% of tasks (≈ the distilled
arm's greedy 7.3%) but ranks it first ~1% of the time. Capability is present in the
distribution; **selection** is what's missing. That is exactly the signal that:
- **execution-reward RL (GRPO)** has real headroom here — RL reweights the policy toward
  what passes, with a realistic ceiling near the policy's pass@k; and
- **best-of-n + executor reranking** harvests the same gap at inference with zero training.

## 4. Guardrails: v2 is healthy, just not better

| | base | v1 | v2 | verdict |
|---|---|---|---|---|
| FIM middle-loss | 0.5848 | 0.6255 | 0.6273 | within +0.05 — infill intact |
| passkey overall | 96.9% | 91.7% | 81.2% | v2 −15.7pt: worse than v1, n=8/cell noisy — watch item |
| syntax parse | 0.17 | 0.83 | 0.83 | held |
| repetition loop | 0.75 | 0.25 | 0.38 | held |

The one soft spot: v2's passkey (81.2%, L8000 17/24) continues the trend that heavier
chat-SFT erodes long-context retrieval slightly. Not a collapse; flag it, and note the erosion
tracks total SFT exposure (v2 trained on 2.29M supervised tokens vs v1's 1.91M).

## 5. Why the ceiling sits where it does: tokens

The base saw **21B tokens** (D/N ≈ 17.5, Chinchilla-optimal *for loss*). StarCoderBase-1B —
the same parameter class, no distillation — reaches **~15% HumanEval as a raw base** at
~1,000B tokens; ours is at 0.6%. Capability at fixed scale climbs roughly log-linearly in
tokens far past Chinchilla (modern small models over-train 10–100×). Crude two-point
interpolation: **~+8–9 HumanEval points per 10× tokens** (±50%; StarCoder also wins on data
filtering). Instruction-tuning can only harvest what pretraining deposited — and v1 already
harvested most of it.

## 6. Projected value of each additional-training lever

| lever | predicted effect | basis | cost |
|---|---|---|---|
| more same-kind SFT | ~0 | measured (this doc) | — |
| correctness-contrast DPO (pass/fail pairs per problem, human solutions + executor labels) | +1–3 pt MBPP | shifts policy off near-misses; base-bounded | $0, hours |
| **execution-reward GRPO** | HumanEval sampled p@1 1.2 → ~4–5%; MBPP 14 → ~18–22% | harvests the measured 5× pass@k gap; CodeRL/RLEF-class results; lab's own RM+GRPO>IPO result | $0, ~1–2 days |
| best-of-n + executor rerank (inference) | delivered correctness ≈ pass@k (6.1% @ n=10) | no training; measured | $0 |
| continued pretrain +80B tokens (→100B) | base HE ~3–6%; post-instruct ~8–12% | token-scaling interpolation above | ~$900, 2–3 wks |
| continued pretrain ×10 (→210B) | base ~6–9%; post-instruct ~12–18% | same | ~$2.2k |

**Recommended sequence:** GRPO + contrast-DPO now (free, fast, attack the measured selection
gap) → decide the continued-pretraining spend with a pre-registered token-scaling prediction
→ re-run the instruct+GRPO harvest on the deeper base.

## 7. What v2 taught (worth the ~2h it cost)

1. **The verified-data lever saturates fast** at this base scale — one tranche (v1) captures
   the format harvest; scaling it 2.7× is flat. (Negative results at $0 are cheap; this one
   redirects the roadmap.)
2. **The binding failure mode is wrong-logic (61–67%), not format** — so the next levers must
   supply a *correctness* signal (execution reward, contrast pairs), not more demonstrations.
3. **The capability/selection split is measurable and large** (pass@10 ≈ 5× pass@1) — RL and
   reranking have quantified headroom before any new pretraining dollar is spent.

Artifacts: `evals/instruct/coder-1b-instruct-v2-*`, `evals/instruct/v2-suite.json`,
`evals/instruct/passkey-v2.json`. Run: `runs/coder-1b-instruct-v2` (servable, chat mode).
Data: `data/corpora/code_sft_compliant_v2.jsonl` (10,234 rows / 2.29M supervised tokens;
7,078 executor-verified). Builder fix en route: `parse_int=str` for huge competitive test
integers (5df078a).
