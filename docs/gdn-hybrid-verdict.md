# GDN/KDA 3:1 hybrid vs dense attention — verdict

`runs/gdn-ab-{dense,hybrid}`, 124M, 4500 steps, seed 1337, FineWeb-100BT. Both arms are the
adopted recipe (Peri-LN + Muon + RoPE + SwiGLU); the only difference is `hybrid_every`
(None vs 4 — three GatedDeltaNet layers per one global attention layer, the published
Kimi Linear ratio).

## Result: LOSS PARITY, with the trend running against the hybrid

| | dense | hybrid | delta |
|---|---|---|---|
| final val loss @4500 | **3.2800** | 3.2824 | **+0.0024** |
| perplexity | 26.58 | 26.64 | |
| params | 109.59M | 115.11M | +5.0% |
| wall-clock | ~1.7 h | ~1.7 h | ~0 |

+0.0024 nats is **exactly at the paired-intervention noise band** (0.0025, see
`periln-verdict.md`). This is parity, not a win and not a loss.

**But the trajectory is the interesting part, and it is not flat.** The hybrid *leads* early
and is overtaken:

| step | 250 | 1000 | 2000 | 2750 | 3500 | 4500 |
|---|---|---|---|---|---|---|
| delta (hybrid − dense) | −0.0080 | −0.0107 | −0.0057 | −0.0001 | +0.0002 | **+0.0024** |

The hybrid's lead peaks around step 750–1000 (−0.011), erodes monotonically, crosses over
near step 2750, and is still widening at 4500. A plausible reading: the linear layers with
decay initialised near 1 impose a strong recency prior that fits fast, while full attention's
advantage on longer-range structure only pays off later. **Extrapolating, the gap would grow
with a longer run** — the opposite of the Peri-LN lane, whose advantage shrank. That is a
caution for scaling this to a real pretrain, and it means "parity at 4500 steps" should not be
read as "parity at 100k steps."

Two confounds, both recorded in the configs before the run:

1. **Params not matched** — the hybrid carries +5.0% (all from GDN's SiLU output gate).
   Param-matched, the hybrid would be *slightly worse* than parity, not equal to it.
2. **Throughput proves nothing here.** Identical wall-clock, but our chunkwise scan has no
   custom kernels. This neither confirms nor refutes the efficiency claim. (It does refute my
   earlier 2.9x-slower estimate, which came from an uncompiled microbenchmark; under
   max-autotune at production batch the two arms are indistinguishable.)

## The axis that decides this lane is memory — now MEASURED, and it validates

Linear attention's premise is not better loss — it is the *same* loss for O(1) state instead
of a KV cache growing linearly in context. Parity on loss is therefore the **expected and
desired** outcome, not a null result. The question is what the parity buys.

Measured 2026-07-30 on the RTX 6000 Ada at the real 124M config, batch 1, bf16, after
building the incremental-decoding path (`HybridCache` + `gdn_step`, gated by
`tests/test_gdn_cache.py` — cached generation is token-for-token identical to uncached, so
these numbers describe the same model):

| context | dense cache | hybrid cache | reduction | dense ms/tok | hybrid ms/tok | latency |
|---|---|---|---|---|---|---|
| 1,024 | 36.7 MB | 11.1 MB | 3.32x | 4.3 | 5.5 | 1.28x slower |
| 4,096 | 150 MB | 39.4 MB | 3.81x | 4.2 | 5.5 | 1.31x slower |
| 16,384 | 603 MB | 153 MB | 3.95x | 4.1 | 5.5 | 1.34x slower |
| 32,768 | 1,206 MB | 303 MB | 3.98x | 4.2 | 5.5 | 1.31x slower |
| 65,536 | 2,414 MB | 605 MB | 3.99x | 5.0 | 5.5 | 1.10x slower |
| **131,072** | **4,830 MB** | **1,209 MB** | **3.99x** | **7.7** | **5.8** | **0.75x — hybrid FASTER** |

**Memory: claim confirmed.** Measured 3.99x against the predicted 4.00x; per-token KV drops
36.0 KB → 9.0 KB. The recurrent state is **1.89 MB and does not move** across a 128x range of
context length (predicted 1.77 MB — the extra 0.12 MB is the depthwise-conv history I forgot
to count). Note the state is kept in fp32 rather than bf16, deliberately: it accumulates
across every token and wants the precision.

**Latency: the O(1) property is visible, and there is a crossover at ~100k.** The hybrid is
*flat at ~5.5 ms/token from 1k to 131k* — that is the architectural claim, measured. Dense is
cheaper below ~65k (4.1–4.2 ms, attention isn't the bottleneck for a 124M model at those
lengths) and then degrades: 4.2 → 5.0 → 7.7 ms. They cross near 100k, and at 131k the hybrid
wins 174 vs 130 tok/s.

**So the honest scope of the win: this architecture is worth it for long context and costs
you ~30% decode latency below ~64k.** Which is exactly why Kimi Linear ships it — they run
128k+. It also means the earlier framing of "parity buys memory for free" was wrong: below
64k the parity buys memory *and costs latency*.

Caveat that cuts in the hybrid's favour: our GDN path has no fused kernels, so the flat
5.5 ms is an implementation artifact, not a floor. With real kernels the constant drops and
the crossover moves substantially earlier. Treat ~100k as a **pessimistic** bound.

## The unexpected result: the hybrid length-generalizes ~10x better than dense attention

Teacher-forced val loss at 1x / 2x / 4x the 1024 training length
(`scripts/eval_length_gen.py`, 300k tokens per length, `evals/length_gen/*.json`):

| arm | L=1024 | L=2048 | L=4096 | Δ at 4x | last-512 drift |
|---|---|---|---|---|---|
| dense NoPE | 3.3504 | 4.2915 | **6.2373** | **+2.887** | **+5.246** |
| dense RoPE | 3.2805 | 3.2457 | 3.4099 | +0.129 | +0.352 |
| hybrid NoPE-globals | 3.2817 | 3.2350 | 3.3448 | +0.063 | +0.191 |
| **hybrid RoPE-globals** | 3.2833 | 3.2331 | **3.2953** | **+0.012** | **+0.039** |

Three things fall out, and the first was not predicted:

1. **The 3:1 hybrid extrapolates an order of magnitude better than dense attention.**
   +0.012 nats at 4x training length vs dense RoPE's +0.129 — and on the last-512-token
   bucket, where extrapolation hurts most, +0.039 vs +0.352 (9x). Mechanistically clean:
   9 of 12 layers are recurrent, have no positional table to run off the end of, and their
   decay is scale-free. **This is a far stronger argument for the architecture than loss
   parity was**, and it compounds with the 4x memory result — both point the same way, at
   long context.
2. **NoPE is rescued by recurrence — the open conditional is CLOSED, affirmatively.** Dense
   NoPE collapses (+2.887, last-bucket +5.246 — the cliff verdict 1 rejected it for). The
   *same* NoPE on the global layers of a hybrid costs +0.063. That is a ~46x smaller
   penalty from nothing but having the linear layers carry position. Verdict 1 stands as
   scoped ("pure NoPE in a dense stack") and is now explicitly bounded.
3. **But RoPE-on-globals still beats NoPE-on-globals at extrapolation** (+0.012 vs +0.063,
   5x), while NoPE-on-globals is a hair better in-window (3.2806 vs 3.2824, inside the
   noise band). So the two are not interchangeable, and the choice depends on the target:
   **keep RoPE on the global layers if long context matters**, which for a coding model it
   does. Kimi Linear's NoPE-globals choice is defensible but is not free here.

Caveat: every number above is one seed, and the in-window differences (3.2805–3.2833) are
all inside the 0.0025 paired band. The *extrapolation* differences are 5–100x that band, so
those are the trustworthy part of this table.

## THE 4500-STEP VERDICT WAS WRONG. At compute-optimal, the hybrid WINS.

The long-run pair (15000 steps, same seed/data/schedule, differing only in `hybrid_every`)
**inverts the 4500-step result**:

| | 4500 steps | 15000 steps |
|---|---|---|
| tokens | 0.74B = **0.30x Chinchilla** | 2.46B = **0.99x Chinchilla** |
| dense | 3.2800 | 3.0602 (ppl 21.33) |
| hybrid | 3.2824 | **3.0544 (ppl 21.21)** |
| delta | **+0.0024 — hybrid behind** | **−0.0059 — hybrid AHEAD** |
| trajectory | led early, crossed over at step 2750, gap widening | **led at all 60 eval points, never crossed** |

The final gap is 2.4x the paired noise band, and it is *stable* through the entire anneal
(−0.0058 … −0.0067 from step 12750 to 15000). There is no crossover.

**My prediction was wrong and it is worth recording why.** I predicted the gap would keep
widening against the hybrid, reasoning that the linear layers' recency prior helps early
while full attention's long-range capacity compounds later. The real explanation for the
4500-step crossover is much duller: **that run stopped at 30% of Chinchilla-optimal.** At
0.30x the ordering had not converged; at 0.99x it is stable and reversed. What looked like
a capacity trend was an under-training artifact.

### The methodological finding, which matters more than this verdict

**Our standard 4500-step ablation protocol trains to 0.30x Chinchilla and can invert its own
verdict.** Same seed, same data, same code — only the schedule length differed, and the
answer flipped sign. That is a far bigger problem than any single lane, because:

- Every verdict produced by the 4500-step protocol is now suspect on the same grounds.
  **This explicitly includes Peri-LN** (`periln-verdict.md`), which won at 4500 steps with an
  advantage that was *decaying monotonically* (−0.069 at step 250 → −0.015 at 4500). A
  decaying advantage measured entirely inside the under-trained regime is exactly the shape
  that inverted here. Peri-LN's adoption should be re-tested at ~15000 steps before it is
  carried into a real pretrain.
- The noise-band work (`periln-verdict.md`) fixed the *denominator* of our ablations. This
  fixes the *duration*. Both were wrong in ways that survived multiple confident writeups.

**New rule: architecture ablations run to >=1x Chinchilla-optimal tokens for the ablation
model, or the verdict is labelled UNDER-TRAINED and is not adoption-grade.** At 124M with
160 seqs x 1024 tokens per step that is ~15000 steps, not 4500. The cost is 3.3x per lane —
which is precisely why cheap lanes must be triaged rather than all run (see the
"do not spend a second seed" note below, which stands).

## Verdict: ADOPT the 3:1 hybrid (superseding the earlier PARITY / ADOPT-CONDITIONAL)

Classification per the verdict-audit protocol: **(iii) fair result at this scale/duration**,
for the loss question. Implementation correctness is gated by `tests/test_gdn.py` (24 tests:
chunkwise vs a sequential reference at fp64 to ~1e-8, fp32 stability under hostile decay,
gradient finiteness, ragged T, causality, and an assertion that the gate init survives
`apply(_init_weights)`).

**Update after the memory measurement:** step 1 below is DONE — the incremental-decoding
path exists and the 4x is measured, not computed. What remains is the long-run check, which
is now the only thing gating this lane.

**The next experiment is not seed 1338.** A second seed would resolve ±0.002 more precisely,
and ±0.002 is decision-irrelevant in both directions — nobody adopts or rejects a linear
hybrid over 0.06 perplexity. Spending 3.5 GPU-hours to sharpen a number that changes nothing
is the same mistake as ablating architecture while the data lanes sit untouched.

What would actually decide it, in priority order:

1. ~~Build the incremental-decoding state path and measure memory/throughput.~~ **DONE
   2026-07-30** — `HybridCache` + `gdn_step`, 3.99x memory confirmed, latency crossover
   measured at ~100k context. See the table above.
2. **Longer-run check — now the gating experiment.** The crossover at ~2750 steps and still-widening gap is the one
   genuine warning sign. Before committing this to a real pretrain, run one arm pair to
   ~15k steps and see whether the gap stabilises or keeps growing.
3. **Param-matched rerun** (drop the output gate) only if 1 and 2 look good — it converts
   "parity with 5% more params" into a clean statement.
4. The **NoPE-on-global-layers arm** (`pos="nope"` + `hybrid_every=4`, the actual Kimi Linear
   config) is now a one-line config change and retests the conditional left open by
   `nope-verdict-audit.md`. Cheap, and interesting independent of the above.

## What this lane already earned

Independent of the verdict, the implementation work paid for itself by finding three bugs
that unit tests alone would not have: a gate init silently clobbered by the model's generic
`apply(_init_weights)` (the same failure class as the 1B's MHA), a `1/A_t` formulation that
overflowed to 1e22 with a learned gate, and an `inf * 0` in the fix for that. All three
produced a *finite* forward pass and only surfaced in training. See `fix/gdn-numerics`.
