# Inference efficiency pass: 4.5× wall-clock for the sampled/best-of workloads

Written 2026-08-12, immediately after the harness milestone. All local, $0, v1 frozen —
this changes HOW candidates are generated, never WHAT the model is.

## The ladder, measured (6 MBPP tasks × n=10, temp 0.7/k40, max_new 400, v1)

| rung | wall | tok/s | peak mem | verdict |
|---|---|---|---|---|
| sequential fp32 (status quo) | 72.4s | 82 | 5.0GB | baseline |
| + same-prompt batching (n as one batch) | 17.1s | 286 | 10.0GB | **4.2×, kept** |
| + bf16 weights | 16.2s | 308 | 5.1GB | **+5% and half the memory, kept** |
| + torch.compile (full model) | — | 971 ms/step | — | rejected: the growing KV slice recompiles every step |
| + compile KDA-only (fixed-shape layers) | — | 12.8 ms/step | — | rejected: no win over eager 11.6ms (cache-object guards) |

Mechanism, established by microbench: the decode step is **kernel-launch-bound, not
bandwidth-bound** at this scale — 11.2ms at B=1 vs 11.6ms at B=10 (batch is nearly
free; that flatness IS the 4× win), but bf16's halved weight reads barely move it, and
the ~2.5ms bandwidth floor is unreachable without a static-shape decode path. FP8 was
not attempted: it only helps a bandwidth-bound regime, which this is not. The remaining
3–4× (11.6ms → ~2.5ms) requires a serve engine with static shapes + CUDA graphs —
filed as future work, not done by flag-flipping.

## Quality gates (pre-stated bands, all PASS)

- Greedy MBPP bf16: **36/257 — identical to fp32** (band ±4).
- Greedy HumanEval bf16: 4/164 vs fp32 5 (band ±2).
- Sampled 60-task MBPP slice, batched bf16: **24/60 delivered vs fp32-sequential 19**
  (band ±6; the batched stream is a different deterministic sampling scheme, so this is
  a distributional gate, not token parity).
- Greedy batched rows are token-for-token identical to `generate_until` — locked by
  tests on dense AND hybrid tiny models (`tests/infer/test_batched_gen.py`).

## What shipped

- `src/microlab/infer/batched.py::generate_batch` — n samples of one prompt as one
  batch; per-row stop-string truncation identical to `generate_until`; shared-generator
  sampling documented as its own deterministic scheme.
- `scripts/eval_code.py --engine batched --dtype bf16` — defaults unchanged
  (sequential/fp32) for continuity with every prior number; the header now records
  engine+dtype, so a result file can never silently mix sampling schemes.
- `scripts/generate_best_of.py` — now batched with stop-early, **bf16 by default**
  (the product CLI gets the full 4.5×; `--dtype fp32` remains).
- `scripts/bench_infer.py` + `evals/bench/` — the benchmark harness and all records.

Practical upshot: a best-of-10 answer now costs ~2× a greedy answer in wall-clock
(was ~9×), at half the serving memory. Best-of-k with tests is now cheap enough to be
the default way to query the model.
