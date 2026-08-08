# coder-1b at step 32,000 — 80%, code val-slice below 1.0

Written 2026-08-08. Step 32,000 of 40,000 (80%); 16.8B of ~21B tokens. `ckpt_32000` on the
mix-v2 val set plus the trainer's val print.

## Loss metrics: monotone descent continues

| metric | 30,000 | 32,000 | Δ |
|---|---|---|---|
| trainer val (block 32k) | 1.2703 | **1.2504** | −0.0199 |
| FIM middle-loss | 0.6132 | **0.6071** | −0.0061 |
| val/code | 1.0082 | **0.9926** | −0.016 |
| val/web | 2.8874 | 2.8553 | −0.032 |
| val/math | 2.1500 | 2.1268 | −0.023 |
| val/markdown | 1.8774 | 1.8601 | −0.017 |
| val/arxiv | 1.1606 | 1.1358 | −0.025 |
| val/commits | 1.0630 | 1.0273 | −0.036 |

Val 1.2504 (ppl 3.49); the **code slice broke below 1.0** (0.9926). Every slice improving.

## Greedy-generation metrics (unchanged noise band)

- Repetition greedy loop_rate 0.75 (6/8); sampled floor: greedy 0.75 → temp0.7 0.21 →
  temp0.8 0.13 → temp1.0 0.04. Same story — noisy decoder property, sampled ~0.
- Syntax parse rate 0.67 (4/6).
- HumanEval 0/164, MBPP 0.0117 (3/257) — both jitter task-to-task at this scale (30k had
  0 / 6; 28k had 2 / 3). Near the floor; no trend to read into the swings.

## Run economics — stability restored

- Banked $146 at step 32,000. After a rough churny/dry-market stretch (dry 4×, data-starved
  India host, dry-spell idles, an aborted A100 test), raising the bid to floor+50%
  (~$1.04/h on the UK 2×PCIE) stopped the churn: the current box has held **~6 h with no
  preemption**. Slow (~20 s/step on 2×PCIE) but steady and cheap.
- Also this stretch: the checkpoint bucket was found bloated to ~1.4 TB by B2 versioning
  retaining pruned checkpoints — purged to 206 GB (live data) + a lifecycle rule so it
  self-maintains. New supervisor guards added: `--min-inet-down` (skip data-starved hosts),
  `--bid-multiplier` (tunable preemption resistance), measured switch-class step-times.
- ~8,000 steps to 40k; ~$30–40 more at these rates.
