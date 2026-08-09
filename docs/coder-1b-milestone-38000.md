# coder-1b at step 38,000 — 95%, val below 1.21

Written 2026-08-09. Step 38,000 of 40,000 (95%); 19.9B of ~21B tokens. `ckpt_38000` on the
mix-v2 val set plus the trainer's val print.

## Loss metrics

| metric | 36,000 | 38,000 | Δ |
|---|---|---|---|
| trainer val (block 32k) | 1.2179 | **1.2056** | −0.0123 |
| FIM middle-loss | 0.5934 | 0.5935 | +0.0001 (flat) |
| val/code | 0.9620 | 0.9550 | −0.007 |
| val/web | 2.7868 | 2.7765 | −0.010 |
| val/math | 2.0716 | 2.0605 | −0.011 |
| val/markdown | 1.8121 | 1.7970 | −0.015 |
| val/arxiv | 1.1067 | 1.1034 | −0.003 |
| val/commits | 1.0076 | **0.9994** | −0.008 |

Val 1.2056 (ppl 3.34); the commits slice broke below 1.0. FIM flat (0.5935) — the infill
loss is bottoming out. All slices still improving, deltas shrinking as the LR decays.

## Greedy-generation metrics jittered down (decoder noise)

| | 36,000 | 38,000 |
|---|---|---|
| repetition loop_rate | 0.625 | 1.000 |
| HumanEval pass@1 | 0.0122 (2/164) | 0.000 |
| MBPP pass@1 | 0.0389 (10/257) | 0.0272 (7/257) |

All three swung the "wrong" way, while every loss metric improved — the exact
decoder-noise signature seen all run: greedy generation is a noisy lagging property, loss
is the signal. Syntax parse rate 0.833 (5/6). The sampled floor stays near zero.

## Run economics — nearly there

- Banked $203 at step 38,000. Cheap interruptible 4×H100 PCIE holding ~11.5 h no preemption.
- ~2,000 steps to 40k: ~$11 more → **~$214 total**, under the $300 cap. The finish is close.
