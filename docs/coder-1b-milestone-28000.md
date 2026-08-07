# coder-1b at step 28,000 — first HumanEval solves

Written 2026-08-07. Step 28,000 of 40,000; 14.7B of ~21B tokens. `ckpt_28000` on the
mix-v2 val set plus the trainer's val prints.

## Headline: the first non-zero HumanEval

| suite | 26,000 | 28,000 |
|---|---|---|
| HumanEval pass@1 | 0.000 (0/164) | **0.0122 (2/164)** |
| MBPP pass@1 | 0.0078 (2/257) | **0.0117 (3/257)** |

The model solved its first HumanEval problems end-to-end under greedy — the first non-zero
reading of the run, after 0/164 at every prior milestone. MBPP ticked up too. Still near
the floor, as expected at this token budget, but the floor is no longer zero: real
execution capability is emerging.

## Loss metrics: clean monotone descent

| metric | 26,000 | 28,000 | Δ |
|---|---|---|---|
| trainer val (block 32k) | 1.3118 | **1.2882** | −0.0236 |
| FIM middle-loss | 0.6512 | **0.6350** | −0.0162 |
| val/code | 1.0497 | 1.0274 | −0.022 |
| val/web | 2.9536 | 2.9156 | −0.038 |
| val/math | 2.2066 | 2.1708 | −0.036 |
| val/markdown | 1.9475 | 1.9007 | −0.047 |
| val/arxiv | 1.1861 | 1.1721 | −0.014 |
| val/commits | 1.1030 | 1.0735 | −0.030 |

Val broke below 1.29 (ppl 3.63). Every slice improving.

## Greedy-generation metrics

- **Syntax parse rate 0.833 (5/6)** — best of the run, up from 0.33 @26k.
- Repetition greedy loop_rate **0.875 (7/8)**, up from 0.75 @26k — still oscillating in
  the 0.6–1.0 band (8-prompt metric, 0.125 quantisation). The stable signal remains the
  sampled floor: decode probe greedy 0.875 → temp0.7 0.21 → temp0.8 0.04 → temp1.0 0.00.
  So the model's distribution keeps improving; greedy loop_rate is a noisy lagging
  indicator that will converge to the floor, not monotonically.

## Run economics

- Banked $128 at step 28,000. Now on a 2×H100 SXM at $1.00/h after the cross-class
  switch-down scanner migrated us off a 4×PCIE box (cheapest cost-per-step among rentable
  options; the 4×SXM market is dry). That migration was also the first production 4→2 GPU
  resume — validated the RNG-restore fix (d725dc9).
- Forecast to 40k on 2×SXM (~11.6 s/step, $1.00/h): ~39h, ~$39 more → ~$167 banked, under
  the $180 cap. The scanner/preemption backstop moves us back to 4×SXM (faster) when the
  market recovers.
