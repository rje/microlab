# coder-1b at step 4,000 — the 2.1B-token milestone, measured

Written 2026-08-05, against the prediction committed in `docs/coder-1b-prediction.md`
BEFORE the first paid step. Training continued past this milestone uninterrupted; all
numbers below are from `ckpt_4000` evaluated locally, on the mix-v2 val set (six slices,
held-out source splits — the set the prediction's thresholds were written for).

## Headline: inside the band, both bold milestones

| step | tokens | predicted mix val | measured | verdict |
|---|---|---|---|---|
| 2,000 | 1.05B | 1.65 – 1.94 | **1.6761** | in band, strong edge |
| 4,000 | 2.10B | 1.49 – 1.76 | **1.5593** | in band, strong half |

Neither falsifier fired at either milestone: never above the broken-run threshold, never
below the leakage tripwire (1.3). Intermediate readings (1.6355 @ 2,500; the per-500
val prints in the shipped log) descend monotonically.

## Per-slice val: every slice improving, none diverging

| slice | step 2,000 | step 4,000 | Δ |
|---|---|---|---|
| code | 1.389 | 1.281 | −0.108 |
| web | 3.554 | 3.367 | −0.187 |
| math | 2.706 | 2.563 | −0.143 |
| markdown | 2.436 | 2.293 | −0.143 |
| arxiv | 1.513 | 1.436 | −0.077 |
| commits | 1.447 | 1.345 | −0.102 |

The per-slice divergence falsifier (one slice degrading behind an improving aggregate)
stays silent. Slice evals run at block 4096 vs the trainer's 32,768, so the slice-implied
aggregate (~1.9) is not comparable to the trainer's 1.5593 — different context length.

## FIM: the format is being learned

Middle-span loss **0.758** (ppl 2.13), n=47/48 scorable. Far below the general code loss
(1.281) because the middle is conditioned on prefix AND suffix — which is the point: the
sentinels steer the model. Only measurable because mix-v2 applies FIM per 4096-token
chunk; under document-level FIM, 73.8% of FIM tokens sat in documents no training window
could contain, and the old eval could score 6 of 64 samples.

## Qualitative: form → shell → proto-semantics

`evals/trajectory/coder-1b-trajectory.md` (frozen prompts, greedy, all milestones).
The binary-search prompt in one line per checkpoint: step 500 dissolves into prose
repetition mid-line; step 2,000 writes syntactically clean function shells with vacuous
bodies; step 4,000 attempts branching logic with real method calls. Syntax parse rate
2/6 → 5/6 across the same interval.

**Watch item — repetition.** Greedy loop rate 0.88 → 1.00. Expected at this depth (the
first 1B looped until far deeper into training), but it is the one non-improving metric.
If loop rate has not begun falling by ~8–10B tokens (steps 16–20k), that is a real
conversation about decoding-side mitigations vs training-side causes.

## Run economics (measured, not projected)

- ~9.3 s/step steady state on 4× H100 PCIE (~56k tok/s aggregate, per-block compile)
- $25.9 spent through step ~4,250, vs $50 originally capped for this milestone
- Full-run projection at these rates: ~$165 total against the raised $180 cap
- 6 preemptions and 2 market droughts to date; every resume from B2 was loss-continuous

## Infrastructure proven by this milestone

Cross-continent resumes (UK/Mexico/Japan hosts), optimizer-state migration, compile-cache
shipping (autotune paid once, not per episode), corpus assertions gating spend, val loss
in the shipped log, milestone checkpoints staged to the Playground, and the fixed-prompt
trajectory sweep — each exists because a specific failure forced it; see git log between
`c657c4a` and `d245110`.
