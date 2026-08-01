# Phase C2 verdict: NoPE retrieval at 32k — DOES NOT PASS

Run: `runs/frontier-32k` (124M, ckpt_15000, 1.60x Chinchilla, block_size 32768,
pos=nope, 3:1 KDA:MLA hybrid). Grid: `evals/passkey-frontier-32k-n64.json`, n=64 per cell.

## Result

Accuracy, passkey depth across the window:

| length | 0.10 | 0.25 | 0.50 | 0.75 | 0.90 |
|---|---|---|---|---|---|
| 1,024 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 4,096 | 1.00 | 1.00 | 1.00 | 0.61 | 1.00 |
| 8,192 | 0.80 | 0.39 | 0.75 | 0.98 | 1.00 |
| 16,384 | 0.00 | 0.00 | 0.30 | **0.98** | 0.27 |
| 32,768 | 0.08 | 0.66 | 0.81 | **0.00** | 0.56 |

**Reliable to 4k. Degrades from 8k. Beyond 16k it is unreliable and non-monotone.**

At n=64 a cell's 95% interval is +/-0.12, so 0.00 vs 0.98 within one length is real, not
sampling noise. The first grid was run at n=8 (+/-0.35) and could not have supported any
verdict; it is kept only as the reason the n=64 run exists.

## It is not a scoring artifact

This repo has had passkey scoring bugs before (`0e432eb`), so the outputs were inspected
rather than trusted. The failure mode is unambiguous:

- success: `" 17827. Remember it."` — the exact key, cleanly
- failure: `" the key that you want to pass the key to the pass"` — **no number at all**

The model does not retrieve a WRONG key; it emits generic filler echoing the prompt's
phrasing, near-identically across samples. The information is absent from the recurrent
state, not corrupted in transit. The eval is measuring what it claims to measure.

## Why C3 passing did not predict this

Length generalisation (`docs/`, `evals/length_gen/frontier-32k.json`) looked healthy:
-0.011 nats beyond the trained window at 1.5x, +0.120 at 2x. That is exactly the trap
flagged when C3 was reported: **a model that ignores everything past position N still
posts good loss, because local context carries almost all of the next-token prediction.**
Loss is necessary and not remotely sufficient. C2 is the test with the teeth, which is why
the plan called it "THE decisive test."

## What this does and does not license

**Does:** the NoPE bet is not validated. We cannot take a globally-NoPE 32k design to the
1B on this evidence.

**Does not:** conclude NoPE is wrong at 1B. The mechanism carrying position is the KDA
recurrent state, whose capacity scales with n_head x head_dim x head_dim. This run is 124M
with head_dim 64; Kimi ships NoPE at 48B. A failure at 124M may be a state-capacity
artifact that does not transfer — the mirror image of the resolution-bias error in
`sota-1b-plan.md` (a null at 124M meaning our test lacks power, rather than the effect
being absent).

Asserting either way from this run alone would repeat the mistake the plan exists to avoid.

## Resolution

A paired A/B at 124M, same seed and data order, 15,000 steps, one field different:

- arm A: `pos="nope"` — already have it (`runs/frontier-32k`)
- arm B: RoPE on the 6 global layers only, KDA layers unchanged

Then the same n=64 passkey grid on both. This is a large-effect test — the class that has
actually decided things in this lab (NoPE +2.887 nats at 4x length; Muon 1.3-1.45x) — not
a marginal loss delta near the noise band. One run, ~35 h locally, and it resolves whether
the 1B ships NoPE or partial-RoPE before any hosted compute is bought.

If arm B also fails past 16k, position is not the binding constraint and the state capacity
is; that would point at head_dim or the KDA:MLA ratio instead, and is worth knowing before
the 1B either way.
