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

## The axis that actually decides this lane is memory, and it is UNMEASURED

Linear attention's premise is not better loss — it is the *same* loss for O(1) state instead
of a KV cache growing linearly in context. Parity on loss is therefore the **expected and
desired** outcome, not a null result. The question is what the parity buys.

Analytically, with only 1 layer in 4 caching K/V:

| context | dense KV | hybrid total | reduction |
|---|---|---|---|
| 1,024 | 37.7 MB | 11.2 MB | 3.37x |
| 4,096 | 151.0 MB | 39.5 MB | 3.82x |
| 32,768 | 1.21 GB | 304 MB | 3.98x |
| 262,144 | 9.66 GB | 2.42 GB | **4.00x** |

Per-token KV drops 36.0 KB → 9.0 KB; the GDN recurrent state is a fixed **1.77 MB**
regardless of context length.

**These numbers are arithmetic, not measurement.** `GatedDeltaNet.forward` deliberately
raises on `kv_cache` — there is no incremental-decoding path, because a recurrent state is not
a KV cache and faking it would silently produce wrong continuations. So the benefit that
justifies the architecture has not been demonstrated on this hardware.

## Verdict: PARITY CONFIRMED / ADOPT-CONDITIONAL — and do NOT spend a second seed on the loss

Classification per the verdict-audit protocol: **(iii) fair result at this scale/duration**,
for the loss question. Implementation correctness is gated by `tests/test_gdn.py` (24 tests:
chunkwise vs a sequential reference at fp64 to ~1e-8, fp32 stability under hostile decay,
gradient finiteness, ragged T, causality, and an assertion that the gate init survives
`apply(_init_weights)`).

**The next experiment is not seed 1338.** A second seed would resolve ±0.002 more precisely,
and ±0.002 is decision-irrelevant in both directions — nobody adopts or rejects a linear
hybrid over 0.06 perplexity. Spending 3.5 GPU-hours to sharpen a number that changes nothing
is the same mistake as ablating architecture while the data lanes sit untouched.

What would actually decide it, in priority order:

1. **Build the incremental-decoding state path** and measure real memory and tokens/sec at
   4k–32k context. This converts the 4x from arithmetic into evidence, and it is the only
   reason to prefer the hybrid at all.
2. **Longer-run check.** The crossover at ~2750 steps and still-widening gap is the one
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
