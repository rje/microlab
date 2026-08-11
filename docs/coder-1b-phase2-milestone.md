# coder-1b Phase 2 milestone: execution-feedback training — two clean negatives, one law

Written 2026-08-11, closing the free-lever program (spec:
`docs/superpowers/specs/2026-08-10-coder-1b-phase2-design.md`; pre-registered bands:
`docs/coder-1b-phase2-prediction.md`, committed before any training). All local, $0.
**Verdict up front: neither training lever beat v1; the pre-registered falsifiers caught both,
each with a diagnosed mechanism. v1 (`runs/coder-1b-instruct-compliant`) remains the lab's
best code model.** The durable outputs are the executor-RL infrastructure, the mechanisms,
and a sharpened picture of where capability actually comes from at 1.2B.

## What was run

1. **Executor-reward GRPO** (`--reward executor`, dense fraction-of-cases rewards, 316
   signal-bearing problems from a 4,311-problem easy-competitive pool, policy pre-passed at
   k=8): 300 iters, β=0.04 (r1); then a controlled retry at β=0.15 with 25-iter segments and
   MBPP-*train* mini-gate checkpoint selection (r2).
2. **Correctness-contrast IPO**: 1,501 executor-labeled pairs (chosen=passing,
   rejected=wrong-output, timeouts excluded), `dpo.py --loss ipo`, 2 epochs from v1.
3. Self-gen was **gated out before training** by its own pre-registered falsifier: v1's
   full-pass yield on the pool measured 7.4% at k=8, under the 15% floor.

## Results vs the pre-registered bands

| arm | MBPP greedy | HumanEval greedy | verdict |
|---|---|---|---|
| v1 (baseline) | 14.0% (36/257) | 3.0% (5/164) | — |
| contrast-IPO | **4.3%** (11/257) | 1.2% (2/164) | **falsifier: below v1 on both** |
| GRPO r1 (β=0.04, 300 it) | **6.2%** (16/257) | 2.4% (4/164) | **falsifier: reward/benchmark divergence** |
| GRPO r2 (β=0.15, best ckpt) | **13.6%** (35/257) | 2.4% (4/164) | flat — no transfer (mini-gate 22–24% across all six checkpoints vs v1's 22%) |

Predicted: GRPO MBPP 18–22%, IPO +1–3. Measured: neither band reached; the "any stage below
v1 on both" and "reward rising while MBPP falls" falsifiers fired exactly as written.

## The mechanisms (each hypothesis was tested, several were killed)

**IPO degenerated.** Training accuracy ended *below chance* (0.20) with loss oscillating in
the thousands — the IPO margin objective (target 1/(2β)) is mis-scaled against
sequence-level logprob gaps on long code responses; the update fought itself and damaged the
policy. Echoes the lab's earlier "naive DPO regressed on the tiny model" result, now at 1.2B
with executor-labeled pairs.

**GRPO r1 over-optimized, and it was not hacking.** Mean group reward rose 0.373→0.507
(windowed, monotone) while MBPP fell 14→6.2 and sampled pass@10 fell 6.1→4.9. Checked and
refuted: reward hacking (top-reward rollouts are genuine passing code), format transfer
(only 1% of MBPP answers used `input()`; completions stayed `def`-style — the v2-era
format-shift hypothesis died a second death). What remained: ~3.8 effective epochs over 316
prompts under a weak KL anchor (drift 0.07→0.8) — the policy overfit the pool and its
general code-logic coherence degraded (failures show scrambled logic, e.g. self-comparison
`if m[i][j] > m[i][j]`). Codex pairwise vs v1 stayed 49/51 — chat *style* survived;
*correctness* is what over-optimization spent.

**GRPO r2 proved the dilemma's other horn.** With β=0.15 the KL stayed bounded (0.02–0.08),
on-pool reward still climbed to ~0.5 — and the mini-gate stayed flat at 22–24% across every
checkpoint (25→150 iters). No damage, no transfer. Together r1/r2 bracket the finding:

> **At 1.2B with a ~300-problem on-policy pool, executor-GRPO either damages the policy
> (weak anchor) or fails to transfer (strong anchor).** The on-pool reward gains are real
> but they are *pool-selection*, not generalizable capability. RL cannot add what
> pretraining didn't deposit; at this scale it can only reweight — and the reweighting
> radius that helps benchmarks is smaller than the radius that overfits a small pool.

## Guardrails (final table)

| | v1 | IPO | GRPO r1 | GRPO r2-best |
|---|---|---|---|---|
| FIM middle-loss (≤0.68) | 0.6255 | n/m (failed gate first) | 0.6348 | 0.6253 |
| passkey (≥82%) | 91.7% | n/m | 85.4% | n/m |
| codex pairwise vs v1 (≥40%) | — | n/m | 49/51 tie | n/m |

("n/m" = not measured: an arm that failed its primary gate wasn't taken through the rest.)

## What Phase 2 actually bought (beyond the negatives)

- **Working executor-RL infrastructure**: the reward oracle, pool builders, pre-pass,
  `--reward executor` mode, mini-gate checkpoint selection — all tested, reviewed, reusable
  the day a stronger base exists. (LFM2.5-class labs end on exactly this stage; the lever is
  right, the base is too small and the pool too narrow.)
- **The self-gen yield measurement** (7.4% < 15% floor) — the loop's precondition is
  quantified, not guessed.
- **A sharpened capability law for the roadmap**: SFT saturates in one tranche (v2 result),
  RL reweights but can't add (this result), the pass@k selection gap is real but must be
  harvested at *inference* (rerank/self-debug), and the floor itself is pretraining tokens.
  Every arrow points the same way: test-time harness now, tokens when/if money is spent.

## Operational fixes that came out of the run

`--easy-only` pool calibration (raw competitive = 0/480 case passes from a 1.2B);
`parse_int=str` for 9,000-digit test integers; batched `sample_solutions`; eval_suite
handling of GRPO checkpoints (VariantConfig cfg → explicit `tokens_seen: null`); the
`ls`-lexicographic vs numeric checkpoint-sort trap in segment scripts.

## Status

v1 stands. The failed arms (`coder-1b-ipo-contrast`, `coder-1b-grpo-exec`,
`coder-1b-grpo-r2*`) are retained on disk as artifacts of record, not promoted. Next
recommended work (user decision pending): the test-time execution-feedback harness
(rerank + a measured repair-probe), where the gains are training-free and already
quantified at ~5× pass@1.

## Addendum (post-merge): the repair-probe

One repair attempt per v1 MBPP failure (task + failed code + stderr → "write a corrected
version"): **1/221 fixed (0.5%)** — runtime errors 0/40, assertions 0/172, syntax 1/8.
The hypothesis that traceback-guided repair might work on the mechanical error class was
**refuted**: at 1.2B the model cannot use error feedback. Design consequence for the
test-time harness: **resample + executor-rerank only** (the measured ~5× pass@k gap);
no repair rung until a stronger base exists.
