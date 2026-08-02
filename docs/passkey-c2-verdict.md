# Phase C2: RETRACTED — the test had no positive control

**Status: this measurement cannot support an architecture verdict. Do not cite the earlier
version of this file, which read the result as evidence against NoPE.**

Run: `runs/frontier-32k` (124M, ckpt_15000, 1.60x Chinchilla, block_size 32768, pos=nope,
3:1 KDA:MLA hybrid). Grid: `evals/passkey-frontier-32k-n64.json`, n=64.

## What was measured

| length | 0.10 | 0.25 | 0.50 | 0.75 | 0.90 |
|---|---|---|---|---|---|
| 1,024 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 4,096 | 1.00 | 1.00 | 1.00 | 0.61 | 1.00 |
| 8,192 | 0.80 | 0.39 | 0.75 | 0.98 | 1.00 |
| 16,384 | 0.00 | 0.00 | 0.30 | 0.98 | 0.27 |
| 32,768 | 0.08 | 0.66 | 0.81 | 0.00 | 0.56 |

The numbers are sound: n=64 (+/-0.12), and the outputs were inspected — successes emit the
exact key, failures emit filler containing no number at all, so it is not a scoring
artifact.

## Why it cannot be read as an architecture verdict

**There is no positive control.** Every passkey result this lab has produced:

| model | params | trained at | result |
|---|---|---|---|
| 1b-4k | 1B | 4,096 | 1.00 across the whole window |
| 1b-4k-v2 | 1B | 4,096 | 0.87–1.00 across 4k |
| 1b-base | 1B | 1,024 | 1.00 inside 1,024, 0.00 outside (the RoPE cliff) |
| frontier-32k | 124M | 32,768 | 1.00 at 1k, degrades from 8k |

Before this run, **no 124M model had ever been passkey-tested here — zero cells.** And no
model at any scale in this lab has ever retrieved beyond 4k; the best long-context result
we own is 1B at 4k.

So the frontier run was asked for retrieval at **8x the context length of anything we have
ever achieved, at 1/8 the parameters**, and its failure was scored against no baseline. The
test cannot separate "NoPE does not carry position" from "a 124M model on 3.93B tokens
cannot retrieve at 32k regardless of position encoding."

Two further reasons to doubt the position reading specifically:

- The failure is **non-monotone in distance** (0.00 at depth 0.25 and 0.98 at 0.75, same
  length). Position-encoding failure degrades smoothly with distance; the 1b-base row above
  is what that actually looks like — a clean cliff at the training window.
- The 6 MLA layers are **full attention over the whole window**. The key is directly
  attendable from the final position at any depth; the recurrence does not have to carry it.

## What the run does establish

- A 124M model **can** do passkey retrieval: 1.00 at every depth at length 1,024.
- Retrieval degrades with length under this training budget. Cause unattributed.
- Distractor density matters more than distance: with real code as filler instead of
  `"The grass is green"`, accuracy collapses at **4k** (0.08–0.17) where English filler
  scores 1.00. The English probe has no digits at all, so the key is the only number in the
  context and is findable by content alone. The two fillers are different tasks.

## Consequence for the plan

`sota-1b-plan.md` names C2 "THE decisive test." It is not decisive as specified, because it
was never powered at the scale it runs at. Options, cheapest first:

1. **Gate at a length with a control.** We know 1k works at 124M. Establishing the
   retrieval frontier as a function of length, with distractor density held fixed, is an
   eval-only measurement on checkpoints we already have.
2. **Buy the control.** A 124M RoPE-on-globals arm at 32k, same seed and data order. Its
   value is *not* "which architecture wins" — it is "is 32k retrieval reachable at all at
   this scale and budget." If RoPE also fails, the test is unpowered and no architecture
   conclusion follows from either arm. ~35 h.
3. **Question the vehicle.** If validating a 32k design needs a model that can retrieve at
   32k, and nothing at 124M can, then 124M may be the wrong scale to validate this design
   at — which is a plan-level finding, not an architecture one.

Whichever is chosen, the NoPE-vs-RoPE decision for the 1B is **unresolved**, and this file
no longer supports either side.
