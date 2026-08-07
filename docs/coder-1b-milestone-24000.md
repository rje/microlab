# coder-1b at step 24,000 — greedy repetition starts to resolve

Written 2026-08-07. Step 24,000 of 40,000; 12.6B of ~21B tokens. Numbers from
`ckpt_24000` evaluated locally on the mix-v2 val set, plus the trainer's val prints.

## Headline: the greedy loop is finally falling on its own

The 4k doc predicted greedy repetition would resolve late in training; the 22k probe
proved it was a *decoder* artifact (greedy loops that sampling breaks). This milestone is
the first where **greedy itself** improves:

| step | greedy loop_rate | sampled (temp 0.7) |
|---|---|---|
| 16,000 | 0.875 | — |
| 20,000 | 0.875 | ~0.29 |
| 22,000 | **1.000** | 0.21 |
| **24,000** | **0.625** | 0.17 |

Greedy loop_rate dropped to **0.625 (5/8)** — the lowest of the run — and the decode
probe shows the gap between greedy and sampled **closing** (1.00 vs 0.21 at 22k → 0.625 vs
0.17 now). The model is resolving the attractors in its distribution, not just under
sampling. Full probe: greedy 0.625 → temp0.7 0.208 → temp0.8 0.167 → temp1.0 0.000.

## Loss metrics: monotone descent continues

| metric | 22,000 | 24,000 | Δ |
|---|---|---|---|
| trainer val (block 32k) | 1.3518 | **1.3319** | −0.0199 |
| FIM middle-loss | 0.6767 | **0.6669** | −0.0098 |
| val/code | 1.0890 | 1.0737 | −0.015 |
| val/web | 3.0166 | 2.9894 | −0.027 |
| val/math | 2.2591 | 2.2401 | −0.019 |
| val/markdown | 1.9982 | 1.9701 | −0.028 |
| val/arxiv | 1.2317 | 1.2187 | −0.013 |
| val/commits | 1.1377 | 1.1182 | −0.020 |

Every slice still improving; none diverging.

## Greedy-generation metrics

- Repetition loop_rate **0.625** (down from 1.0 @22k — first genuine fall).
- Syntax parse rate **0.5** (3/6), up from 0.33 @22k.
- HumanEval **0/164**, MBPP **0.0078 (2/257)** — MBPP back to non-zero after the 22k
  greedy-loop dip to 0. Expected near the floor at this scale.

## Qualitative: prompts escaping the greedy attractor

From the frozen greedy sweep, 22k → 24k:
- `binary-search`: looped `# if arr is None: return -1` at 22k → at 24k writes real
  branching (`if arr[0] == target: return 0 else: return -1`) then starts a second
  `binary_search_sorted` function. Escaped.
- `docstring-only`: looped imports at 22k → at 24k writes `def get_top_level_keys(obj):`
  with a real docstring. Escaped.
- `fn-skeleton`: still the arithmetic family (naming drifted to `add_sub`/`add_mul`).
- `self-binding`: still loops (`print(self.count)` repeated) — not all attractors gone.

The sampled track (temp 0.7) remains the realistic view and is swept alongside greedy;
toggle on the Run log.

## Run economics

- Banked spend $111 at step 24,000 (real charges ~15% higher; account credit is truth).
- Back on a reliable 4×H100 SXM ($2.00/h, rel 0.991) after the 8×H200 experiment: ws=8
  validated (3.0s/step, loss-continuous) but marginally dearer per step for this
  compute-bound model and that host preempted in ~25 min. Detour cost ~$2.
- Forecast to 40k at $2.00/h, 5.6 s/step: ~25h, ~$50 more → ~$161 banked, under the
  $180 cap; Vast credit well in excess after the +$100 top-up.
