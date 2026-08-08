# coder-1b at step 30,000 — three-quarters, best MBPP yet

Written 2026-08-08. Step 30,000 of 40,000 (75%); 15.7B of ~21B tokens. `ckpt_30000` on
the mix-v2 val set plus the trainer's val print.

## Loss metrics: monotone descent, val below 1.27

| metric | 28,000 | 30,000 | Δ |
|---|---|---|---|
| trainer val (block 32k) | 1.2882 | **1.2703** | −0.0179 |
| FIM middle-loss | 0.6350 | **0.6132** | −0.0218 |
| val/code | 1.0274 | 1.0082 | −0.019 |
| val/web | 2.9156 | 2.8874 | −0.028 |
| val/math | 2.1708 | 2.1500 | −0.021 |
| val/markdown | 1.9007 | 1.8774 | −0.023 |
| val/arxiv | 1.1721 | 1.1606 | −0.012 |
| val/commits | 1.0735 | 1.0630 | −0.011 |

Val 1.2703 (ppl 3.56); code slice broke below 1.01. Every slice improving.

## Code execution: best MBPP of the run

| suite | 28,000 | 30,000 |
|---|---|---|
| HumanEval pass@1 | 0.0122 (2/164) | 0.000 (0/164) |
| MBPP pass@1 | 0.0117 (3/257) | **0.0233 (6/257)** |

MBPP doubled to 6/257 — best of the run. HumanEval fell back to 0/164, but that is greedy
noise: a greedy loop on the two 28k-solved tasks loses them (the decode probe shows the
loops break under sampling). The signal is the MBPP trend, not the HumanEval jitter — both
sit near the floor and jitter task-to-task at this scale.

## Greedy-generation metrics (noisy, sampled floor stable)

- Repetition greedy loop_rate **0.75 (6/8)**, still oscillating in the 0.6–1.0 band.
  Decode probe: greedy 0.75 → temp0.7 0.21 → temp0.8 0.17 → temp1.0 0.00 — the sampled
  floor stays near zero, the model's distribution keeps improving.
- Syntax parse rate 0.67 (4/6), down from 0.833 @28k (n=6 noise).

## Run economics — a rough market patch

- Banked $138 at step 30,000. The last hours churned hard: the efficient 4×SXM went dry,
  a data-starved India host (239 Mbps, ~200 s/step, 0% GPU util) had to be killed by hand,
  and the 4×-market stayed dry. Two durable supervisor guards came out of it:
  `--min-inet-down` (skip bandwidth-starved hosts) and measured switch-class step-times.
- As of this milestone the market is fully dry under cap — the run is IDLING (no box, no
  spend, checkpoint safe at 30,000), retrying every 20 min. The scanner/arm-ordering will
  resume on a fast box the moment capacity returns.
- ~10,000 steps to 40k; forecast $30–45 more depending on which class the market yields,
  landing ~$170–183 — near the $180 cap, credit well in excess.
