# coder-1b Phase 2: the free-lever program — design

Written 2026-08-10, after the v2 negative result (`docs/coder-1b-instruct-v2-analysis.md`)
established that more SFT data is flat and quantified the remaining levers: a 5× selection
gap (sampled pass@1 1.2% vs pass@10 6.1% on HumanEval) and a wrong-logic failure ceiling
(61–67% of failures are clean code with wrong output).

## Goal

Make `coder-1b-instruct` the best 1B it can be with **$0 spend** (all local, RTX 6000 Ada),
using execution-feedback training and the self-gen loop. **Headline success criterion: a
fully compliant model that beats the distilled reference on BOTH benchmarks** (distilled ref:
14.8% MBPP / 7.3% HumanEval, greedy pass@1). Base for all training: `runs/coder-1b-instruct-
compliant` (v1, the canonical compliant arm).

## Global constraints

- **Build capability, don't distill.** The executor (`microlab.evals.code.executor.run_python`)
  is the only reward/filter ground truth. External models act ONLY as judges: the **codex CLI**
  (as in `eval_pairwise.py` / `rlaif_judge.py` — schema-forced, position-swapped) for
  chat-quality regression checks. No teacher text enters training.
- **Pre-registered predictions before each training stage** (the lab discipline), scored after.
- **Guardrails after every stage:** FIM middle-loss (≤ ~0.05 over the 0.5848 base), passkey
  long-context (watch item — v1 91.7%, v2 81.2%), repetition, plus a codex pairwise check vs
  v1 on held-out instructions (execution-RL must not wreck general instruction-following;
  mild reward-hacking was caught by telemetry on the prose run — watch for it here).
- **No silent fallbacks; verify by count** (builders/pools report sizes; zero-signal cases
  fail loudly or are counted, never masked).
- Decontamination: any new training prompts (pool, self-gen) pass the same n=10-gram filter
  against HumanEval/MBPP as Phase 1, identically at every stage.

## Architecture — four units, stacked with eval gates

### Unit 1 — Executor reward oracle + prompt pool (shared foundation)

- `executor_score_texts(problem) -> score_texts(prompt, texts) -> list[float]`: matches
  `run_grpo`'s existing oracle interface (the RM's `make_score_texts` is the precedent).
  Pipeline per rollout: truncate at the `### End` sentinel → extract the code block (reuse
  `evals/code/prompts.py` chat extraction) → assemble with the problem's I/O cases
  (`assemble_io_program`) → `run_python` → **reward = fraction of checked cases passed**
  (dense; binary all-pass would zero-out most groups at ~14% pass rates). No code block
  extractable → reward 0.
- **Prompt pool**: problems + I/O cases from the Phase-1 verify pipeline (APPS/CodeContests/
  TACO adapters + MBPP-train), written as JSONL `{instruction, io: [{input, output}, ...]}`.
  Decontaminated. Capped cases per problem (the Phase-1 sanity-filter setting).
- **Policy pre-pass**: sample k=8 from v1 per pool problem, record per-problem success rate;
  keep problems with 0 < successes < k (signal-bearing: mixed groups). All-fail and all-pass
  problems carry zero group advantage. Report pool sizes before/after filtering.

### Unit 2 — Executor-GRPO (flagship)

- `train_grpo` gains `--reward executor --pool <jsonl>` (RM path untouched; the reward oracle
  is swapped behind the same `score_texts` interface). Rollouts already use the chat template
  and stop sentinel.
- Trained from v1. Memory: policy + frozen ref at 1.2B — proven locally by the prose 1b-grpo;
  no RM loaded (the executor replaces it).
- **Pre-registered prediction (commit before training):** MBPP greedy 14.0 → **18–22%**;
  HumanEval sampled pass@1 1.2 → **4–5%** (ceiling: the 6.1% pass@10). Falsifiers: any
  benchmark below v1; FIM/passkey guardrail breach; codex pairwise vs v1 worse than ~40/60.

### Unit 3 — Correctness-contrast DPO (independent measurement)

- Builder change: the verify pipeline keeps **wrong-output failures** (assertion-style — NOT
  timeouts, a slow-but-correct solution is a bad "rejected") alongside the passing solution,
  emitting `{prompt, chosen, rejected}` pairs per problem.
- Train with `dpo.py --loss ipo` from v1 (lab precedent: IPO over naive DPO at small scale),
  **independently of GRPO** so each lever's effect is cleanly attributable.
- Prediction: +1–3 MBPP points over v1; same falsifier structure.

### Unit 4 — Best-of-n rerank eval + self-gen loop (last, from the best checkpoint)

- **(a) Rerank eval**: k-sample + executor-rerank on the benchmarks → the *delivered
  correctness* number (measures the pass@k harvest; ~6% HumanEval at n=10 today). Eval-only:
  real requests carry no tests, so this is measurement + data machinery, not a serve feature.
- **(b) Self-gen**: best-so-far model samples k per pool problem (the FULL pool, not just
  signal-bearing), executor keeps passers, dedup + decontaminate → self-gen SFT tranche →
  fine-tune from the best checkpoint → eval. Fully compliant (own model + executor labels).
  Run last: yield scales with the pass rate the earlier units buy.

### Stacking logic

1. Unit 2 (GRPO) and Unit 3 (DPO) both train **from v1**, evaluated separately.
2. If both beat v1: stack — re-run the lever with the SMALLER solo gain, training from the
   bigger winner's checkpoint (same hyperparameters as its solo run); eval the stack and keep
   it only if it beats the solo winner. If only one beats v1, that one is the winner; if
   neither does, v1 stands and Unit 4 runs from v1.
3. Unit 4 self-gen runs from the best checkpoint after stacking; final eval + guardrails +
   codex pairwise.
4. Milestone doc scores every prediction and names the final model (the "best 1B" deliverable).

## Evaluation battery (every gate)

HumanEval + MBPP greedy pass@1 (primary; the honest metric post-SFT), HumanEval sampled
pass@1/@10 (selection-gap tracking), error-mode taxonomy (the wrong-logic share should FALL
if the levers work), FIM + passkey + repetition guardrails, codex pairwise vs v1 (held-out
`code_sft_heldout.jsonl`, 120 prompts, both orderings).

## Risks

- **Zero-signal collapse:** if the pre-pass leaves too few signal-bearing problems, GRPO
  starves. Mitigation: dense per-case rewards widen the mixed-group set; pool size is
  reported before training starts (verify by count) and the run doesn't launch under a floor.
- **Reward hacking:** printing expected outputs verbatim, hardcoding cases. Mitigation:
  multiple I/O cases per problem, telemetry on reward-vs-benchmark divergence, the codex
  pairwise check, and eyeballing top-reward rollouts at each checkpoint (the prose run's
  hashtag-hacking was caught exactly this way).
- **Long-context erosion:** passkey has slid v1→v2 with SFT exposure; every stage gates on it.
- **Self-gen echo:** training on own outputs can amplify quirks; mitigation: executor filter,
  dedup, cap the self-gen tranche relative to the human data, and the guardrail battery.

## Deliverables

1. `src/microlab/train/exec_reward.py` (oracle + pool: pure, testable) + `--reward executor`
   in `scripts/train_grpo.py`.
2. Pool + pre-pass builder (`scripts/build_grpo_pool.py`), contrast-pairs builder mode,
   self-gen builder (`scripts/build_selfgen_sft.py`).
3. Rerank eval (`scripts/eval_rerank.py`).
4. `docs/coder-1b-phase2-prediction.md` (pre-registered, before any training).
5. Trained checkpoints per stage; final named model; milestone doc scoring all predictions.
