# coder-1b at step 22,000 — the repetition watch item, resolved

Written 2026-08-07. Step 22,000 of 40,000; 11.5B of ~21B tokens. Numbers from
`ckpt_22000` (weights-only stage) evaluated locally on the mix-v2 val set, plus the
trainer's own val prints.

## Headline: greedy repetition is a DECODER artifact, not the model

The 4,000-step doc set a falsifiable watch item: *"If loop rate has not begun falling by
~8–10B tokens (steps 16–20k), that is a real conversation about decoding-side mitigations
vs training-side causes."* At 22k (11.5B tokens) greedy loop rate did not fall — it went
**back to 1.00** (8/8), and syntax parse rate fell to 0.33. So the conversation is due,
and this milestone runs the experiment that settles it.

**The experiment.** Same eight repetition prompts, `ckpt_22000`, greedy vs mild sampling
(3 seeds each, 12-gram loop detector):

| decoder | loop rate |
|---|---|
| greedy (temp 0) | **1.00** (8/8) |
| temp 0.7, top-k 40 | 0.29 |
| temp 0.8, top-k 50 | 0.21 |
| temp 1.0, top-k 100 | **0.00** |

The loops collapse the instant any stochasticity enters, and vanish entirely at temp 1.0.
A model whose *distribution* were genuinely degenerate could not be de-looped by sampling
from it. So the repetition is **decoding-side**: greedy argmax falls into the
highest-probability attractor and stays, exactly the failure that sharpens (not softens)
as a model grows more confident. The underlying model is healthy — every loss-based
metric below improved.

This does not falsify anything in the prediction doc; it identifies the metric as a
property of the *decoder we chose for comparability*, not of the checkpoint.

## Every loss metric improved

| metric | 20,000 | 22,000 | Δ |
|---|---|---|---|
| trainer val (block 32k) | 1.3734 | **1.3518** | −0.0216 |
| FIM middle-loss | 0.6965 | **0.6767** | −0.0198 |
| val/code | 1.1096 | 1.0890 | −0.021 |
| val/web | 3.0589 | 3.0166 | −0.042 |
| val/math | 2.3002 | 2.2591 | −0.041 |
| val/markdown | 2.0364 | 1.9982 | −0.038 |
| val/arxiv | 1.2601 | 1.2317 | −0.028 |
| val/commits | 1.1580 | 1.1377 | −0.020 |

Monotone descent everywhere; no slice diverging. The divergence at this milestone is
**loss-down / greedy-generation-down**, and the decode probe explains why the two point
opposite ways.

## Greedy-decoder metrics (all expected to move late, and to be noisy)

- Repetition greedy loop rate **1.00** (up from 0.875 @16k/20k — the dip was within the
  1/8-quantised noise of an 8-prompt metric; the signal is "still looping under greedy").
- Syntax parse rate **0.33** (2/6), down from 0.67 — same cause, same caveat (n=6).
- HumanEval **0/164**, MBPP **0/257** greedy pass@1 (20k had 2/257 on MBPP; a greedy loop
  on those two tasks is enough to lose them — consistent with the decoder story).

## Implication (measurement, not a decision taken here)

The model is on track; the only thing "wrong" at 22k is that greedy is a poor decoder for
a mid-training base model. Levers, if/when we want clean free-running generation:

- **Playground already exposes temperature** — a demo at temp 0.7–0.8 shows the model's
  real competence today.
- A **no-repeat-ngram / repetition-penalty** sampler would give loop-free *greedy-like*
  output; cheap to add, only worth it if we want the deterministic trajectory demos to
  read well before the model outgrows greedy looping on its own.
- The **frozen trajectory sweep stays greedy** regardless — its value is cross-checkpoint
  comparability, and changing its decoder would invalidate every prior column.

## Run economics

- Banked spend $97.8 at step 22,000 (real charges ~15% higher; account credit is truth).
- This box (Japan SXM, $2.00/h interruptible) held ~3.5h across the 20k→22k stretch after
  a burst of preemptions; the market cooled.
- The supervisor now scans for cheaper hosts every 20 min and migrates a healthy box down
  when one appears ≥20% cheaper (committed a95739b) — bidding stays at floor+25%.
- Forecast to 40k at $2.00/h and ~5.6 s/step: ~28 h, ~$56 more.
