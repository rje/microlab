# coder-1b at step 26,000 — loss descends; greedy repetition oscillates

Written 2026-08-07. Step 26,000 of 40,000; 13.6B of ~21B tokens. `ckpt_26000` on the
mix-v2 val set plus the trainer's val prints.

## Loss metrics: clean monotone descent (the reliable signal)

| metric | 24,000 | 26,000 | Δ |
|---|---|---|---|
| trainer val (block 32k) | 1.3319 | **1.3118** | −0.0201 |
| FIM middle-loss | 0.6669 | **0.6512** | −0.0157 |
| val/code | 1.0737 | 1.0497 | −0.024 |
| val/web | 2.9894 | 2.9536 | −0.036 |
| val/math | 2.2401 | 2.2066 | −0.034 |
| val/markdown | 1.9701 | 1.9475 | −0.023 |
| val/arxiv | 1.2187 | 1.1861 | −0.033 |
| val/commits | 1.1182 | 1.1030 | −0.015 |

Every slice improving, none diverging. Val is now 1.3118 (ppl 3.71).

## Greedy repetition: the 24k drop was partly noise

| step | greedy loop_rate | sampled (temp 0.7) |
|---|---|---|
| 20,000 | 0.875 | ~0.29 |
| 22,000 | 1.000 | 0.21 |
| 24,000 | **0.625** | 0.17 |
| 26,000 | **0.750** | 0.21 |

Greedy loop_rate rose back to **0.75 (6/8)** from 0.625 at 24k. On an 8-prompt metric
(quantised to 0.125) this is within noise — the honest read is that greedy loop_rate is
**oscillating in the 0.6–1.0 band with at most a weak downward tendency**, not cleanly
resolving. The 24k milestone doc over-read a single low sample; this corrects it. What is
stable is the **sampled floor** (decode probe: greedy 0.75 → temp0.7 0.21 → temp0.8 0.08
→ temp1.0 0.04), confirming again that the model's distribution is fine and greedy is the
pathology. Syntax parse rate 0.33 (n=6, same noise caveat).

The lesson stands from the 22k probe: read loss/FIM/per-slice as the real trajectory and
treat the greedy generation metrics as a noisy decoder property that will converge to the
sampled floor eventually — just not monotonically, and not yet decisively.

## Code execution

HumanEval **0/164**, MBPP **0.0078 (2/257)** — steady at the floor, as expected at this
token budget.

## Run economics

- Banked $119 at step 26,000. Back on reliable 4×H100 SXM ($2.00/h) after a clean
  preemption recovery at 24,400 (≤50 steps lost).
- Forecast to 40k at $2.00/h, ~5s/step: ~20h, ~$40 more → ~$159 banked, under the $180
  cap; credit well in excess.
