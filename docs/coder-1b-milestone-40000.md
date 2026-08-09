# coder-1b at step 40,000 — the run is complete

Written 2026-08-09. **Step 40,000 of 40,000 (100%)**; 20.97B of ~21B tokens. `ckpt_40000`
on the mix-v2 val set, the trainer's block-32k val print, and the full delegated battery
(HumanEval, MBPP, probes). This is the capstone milestone: the final checkpoint, measured,
against the prediction committed in `docs/coder-1b-prediction.md` *before the first paid
step*.

## Headline: finished at val 1.1993, every falsifier silent for the whole run

The run reached target on an on-demand UK 4×H100 PCIE (the interruptible market went dry at
97.6%), destroyed the box on target-reached, and left zero billing instances. Total spend
**~$227**. Not one of the prediction's four falsifiers fired at any point across 40,000
steps.

| final metric | value | vs 38,000 |
|---|---|---|
| trainer val (block 32k) | **1.1993** (ppl 3.32) | −0.0063 |
| FIM middle-loss | **0.5848** (ppl 1.79), n=63 | −0.0087 (resumed descent after the 38k flat; best of run) |
| HumanEval pass@1 | 0.0061 (1/164) | greedy noise band (0–2/164 all run) |
| MBPP pass@1 | **0.0389 (10/257)** | ties the 36k run-best |

## Loss trajectory: monotone from warmup to step 40,000

| step | tokens | trainer val (32k) | FIM mid | | step | tokens | trainer val (32k) | FIM mid |
|---|---|---|---|---|---|---|---|---|
| 2,000 | 1.05B | 1.6761 | 0.758 | | 30,000 | 15.7B | 1.2703 | 0.6132 |
| 4,000 | 2.10B | 1.5593 | 0.758 | | 32,000 | 16.8B | 1.2504 | 0.6071 |
| 20,000 | 10.5B | 1.3734 | 0.6965 | | 34,000 | 17.8B | 1.2328 | 0.6019 |
| 22,000 | 11.5B | 1.3518 | 0.6767 | | 36,000 | 18.9B | 1.2179 | 0.5934 |
| 24,000 | 12.6B | 1.3319 | 0.6669 | | 38,000 | 19.9B | 1.2056 | 0.5935 |
| 26,000 | 13.6B | 1.3118 | 0.6512 | | **40,000** | **20.97B** | **1.1993** | **0.5848** |
| 28,000 | 14.7B | 1.2882 | 0.6350 | | | | | |

Strictly monotone after the 700-step warmup, the descent flattening only as the LR decayed
to ~0 (38k→40k just −0.006). Final val 1.1993 (ppl 3.32); FIM 0.5848 (ppl 1.79).

## Per-slice val: every slice at a run-best, none diverging

| slice | 38,000 | 40,000 | Δ |
|---|---|---|---|
| code | 0.9550 | **0.9492** | −0.006 |
| web | 2.7765 | **2.7722** | −0.004 |
| math | 2.0605 | **2.0505** | −0.010 |
| markdown | 1.7970 | **1.7842** | −0.013 |
| arxiv | 1.1034 | **1.0938** | −0.010 |
| commits | 0.9994 | **0.9935** | −0.006 |

Every slice is at its lowest of the run. The per-slice divergence falsifier — one slice
degrading behind an improving aggregate — stayed silent at **every** milestone from 2,000
to 40,000. (Slice evals run at block 4096 vs the trainer's 32,768, so the slice-implied
aggregate is not directly comparable to the trainer's 1.1993.)

## Prediction scorecard: 0 of 4 falsifiers fired

The prediction (written before the run) named four ways to be wrong. None occurred:

1. **Above 2.2 at step 2,000** (broken run) — no: measured 1.6761, inside the 1.65–1.94 band.
2. **Below 1.3 at step 2,000** (leakage tripwire) — no: 1.6761, well above; the mix builder
   was clean.
3. **Non-monotone val after warmup** (instability) — no: strictly monotone for 40,000 steps.
4. **Per-slice divergence** — no: every slice improved at every milestone.

Both bold early milestones landed in-band (2,000: 1.6761 ∈ [1.65, 1.94]; 4,000: 1.5593 ∈
[1.49, 1.76]). The prediction table stopped at step 7,629 (4B tokens, the "$20 run"); the
run continued to 21B tokens, so the tail is beyond the predicted band, but the anchor logic
holds: at 21B tokens the prose-only 1b anchor sat at 2.501, and this 67.5%-code-like mix
finished far below it at 1.1993 — code is more predictable, exactly as the mix reasoning
assumed.

## Compute regime: finished near Chinchilla-optimal

At 20.97B tokens on 1.20B parameters, the finished model sits at **D/N ≈ 17.5** — close to
Chinchilla's compute-optimal 20, and a genuine shift from the early-run framing where the
prediction called the model "deeply under-trained (D/N ≈ 1–4)." The last checkpoint is the
first that is *not* materially under-trained, which is why the loss curve is bottoming
rather than still falling steeply.

## Code execution: at the scale's floor, but the floor climbed

| suite | 34k | 36k | 38k | 40k |
|---|---|---|---|---|
| HumanEval pass@1 | 0.000 | 0.0122 (2/164) | 0.000 | 0.0061 (1/164) |
| MBPP pass@1 | 0.0272 (7/257) | **0.0389 (10/257)** | 0.0272 (7/257) | **0.0389 (10/257)** |

MBPP finished at 10/257, matching the 36k run-best; HumanEval jitters in a 0–2/164 band all
run. These are ceiling measurements for a compute-optimal 1.2B base model with no
instruction tuning — near the floor, as predicted, with `syntax_valid` and the loss curve
carrying the early signal instead. Reading the HumanEval jitter as regression would be the
same mistake the prediction warned against.

## Decoder-side metrics stayed noisy to the end (as established)

Greedy syntax parse rate **0.17 (1/6)** and repetition loop_rate **0.75 (6/8)** are the same
noisy decoder-side properties seen all run, and they diverge from loss by construction. The
40k syntax "failures" are illustrative: the fibonacci sample is clean, correct Python that
was simply cut mid-line at the token budget — a truncation artifact, not a malformed model.
Greedy argmax loops on this class of base model (loop_rate 1.0 greedy → ~0.0 at temp 1.0 on
earlier probes); the sampled trajectory track is the realistic free-running view. This is
the resolution of the milestone-4,000 watch item ("if loop rate has not begun falling by
16–20k, that is a real conversation"): it was a decoder artifact, not a training pathology —
loss, FIM, and per-slice were the real signal throughout.

The probe battery agrees: in-context analogy works (`apple:fruit … sparrow:` → `bird`, and
the pattern continues sensibly), while factual recall loops (`capital of France` wrong) and
arithmetic is at chance (`8 + 6` → `13`). Reasoning stayed at chance, consistent with the
first 1B — this run bought representation quality (loss/FIM/slices), not emergent reasoning
at this scale and token budget.

Qualitative trajectory (frozen prompts, all milestones, greedy **and** sampled):
`evals/trajectory/coder-1b-trajectory.md` and `-trajectory-sampled.md`.

## Run economics — final reconciliation

- **Total spend ~$227.08**, against a pre-run projection of ~$165 and a cap raised
  180→250→300 over the run. Credit was never the constraint.
- The ~$60 over the original projection was the finish, not the steady state: the
  interruptible market dried up repeatedly in the last 3% (a data-starved India host, an
  8×H100 on-demand box that hung on NCCL init ~$12, dry-market idles), and the last ~950
  steps ran on an on-demand UK 4×H100 PCIE at $8/h because the interruptible market could
  not be caught at 97.6% — a deliberate ~$20 trade of money for a guaranteed finish.
- Steady-state economics held all run: ~9.3 s/step on 4×H100 PCIE, per-block compile,
  every resume from B2 loss-continuous across preemptions and GPU-count changes (including
  the first real 4→2 GPU RNG-restore resume).

## What the run establishes

A 1.2B-parameter code-focused model, trained from scratch on a self-built 21B-token mix, to
a near-compute-optimal final checkpoint, on rented interruptible GPUs under a spend cap —
with a falsifiable loss prediction written first and met end-to-end. The infrastructure that
made it survivable (B2 checkpoint mirroring, cross-continent/cross-GPU-count resume,
compile-cache shipping, cost-per-step switch-down, corpus assertions gating spend) is the
durable output alongside the weights. Next levers for capability are instruction/chat tuning
on top of this base and, per the standing preference, building capability rather than
distilling it.
