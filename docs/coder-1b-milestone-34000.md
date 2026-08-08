# coder-1b at step 34,000 — 85%, best MBPP of the run

Written 2026-08-08. Step 34,000 of 40,000 (85%); 17.8B of ~21B tokens. `ckpt_34000` on the
mix-v2 val set plus the trainer's val print.

## Loss + code: both at run bests

| metric | 32,000 | 34,000 | Δ |
|---|---|---|---|
| trainer val (block 32k) | 1.2504 | **1.2328** | −0.0176 |
| FIM middle-loss | 0.6071 | **0.6019** | −0.0052 |
| val/code | 0.9926 | 0.9764 | −0.016 |
| val/web | 2.8553 | 2.8169 | −0.038 |
| val/math | 2.1268 | 2.0910 | −0.036 |
| val/markdown | 1.8601 | 1.8253 | −0.035 |
| val/arxiv | 1.1358 | 1.1198 | −0.016 |
| val/commits | 1.0273 | 1.0124 | −0.015 |

Val 1.2328 (ppl 3.43); all slices improving.

## Code execution: MBPP best of the run

| suite | 32,000 | 34,000 |
|---|---|---|
| HumanEval pass@1 | 0.000 | 0.000 (0/164) |
| MBPP pass@1 | 0.0117 (3/257) | **0.0272 (7/257)** |

MBPP up to 7/257 — best yet. HumanEval still 0/164 (jitters task-to-task; the greedy loop
costs the handful of solvable ones). Syntax parse rate 0.833 (5/6), also a run best.
Repetition greedy 0.75 (unchanged noise band; sampled floor ~0).

## Run economics — reverted from on-demand to cost-controlled

- Banked $183 at step 34,000. Tried an on-demand finish (8×H100 hung on NCCL in setup;
  4×H100 SXM on-demand worked but at ~2.5–3× the per-step cost) — reverted to a cheap
  interruptible 4×H100 PCIE (UK, $2.08/h at floor+50%). On-demand detour cost ~$24.
- Forecast to 40k on the 4×PCIE (~9.3 s/step, $2.08/h): ~16h, ~$32 more → **~$215 total**,
  under the $300 cap; ~6,000 steps left.
