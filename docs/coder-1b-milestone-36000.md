# coder-1b at step 36,000 — 90%, MBPP into double digits

Written 2026-08-09. Step 36,000 of 40,000 (90%); 18.9B of ~21B tokens. `ckpt_36000` on the
mix-v2 val set plus the trainer's val print.

## Loss metrics

| metric | 34,000 | 36,000 | Δ |
|---|---|---|---|
| trainer val (block 32k) | 1.2328 | **1.2179** | −0.0149 |
| FIM middle-loss | 0.6019 | **0.5934** | −0.0085 |
| val/code | 0.9764 | 0.9620 | −0.014 |
| val/web | 2.8169 | 2.7868 | −0.030 |
| val/math | 2.0910 | 2.0716 | −0.019 |
| val/markdown | 1.8253 | 1.8121 | −0.013 |
| val/arxiv | 1.1198 | 1.1067 | −0.013 |
| val/commits | 1.0124 | 1.0076 | −0.005 |

Val 1.2179 (ppl 3.38); FIM broke below 0.60. All slices improving.

## Code execution: MBPP into double digits

| suite | 34,000 | 36,000 |
|---|---|---|
| HumanEval pass@1 | 0.000 | **0.0122 (2/164)** |
| MBPP pass@1 | 0.0272 (7/257) | **0.0389 (10/257)** |

MBPP reached **10/257** — first double-digit count, best of the run — and HumanEval is
non-zero again (2/164). Both execution metrics up together at this milestone. Still near
the floor for the scale, but the floor is climbing.

Repetition greedy loop_rate **0.625** (down from 0.75; lowest since 24k). Syntax parse rate
0.5 (3/6, noise band). The greedy metrics remain noisy; the loss + MBPP trends are the
signal.

## Run economics

- Banked $193 at step 36,000. Cheap interruptible 4×H100 PCIE (UK, $2.08/h at floor+50%)
  has held ~6.5 h with no preemption.
- Forecast to 40k (~9.3 s/step): ~10h, ~$21 more → **~$214 total**, under the $300 cap.
  ~4,000 steps left.
