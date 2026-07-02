> **Exercise — on `main`, no branch switching.** Implement the stubs in
> `src/microlab/exercises/phase06_inference.py`, then run
> `pytest tests/exercises/test_phase06_inference.py -m exercise` to grade them.

# START HERE — inference engineering (Phase 6)

Everything between a checkpoint and a served token. Four hand-writes, graded against
`microlab.infer.reference`:

1. **`generate_cached`** — the KV cache. Without it, token T recomputes K/V for all T−1
   predecessors (generation is O(T²)); with it, each step is one single-token forward.
   Graded by EXACT token-match against the uncached reference + a measured speedup. This
   is the sharpest test in the curriculum: off-by-one RoPE offsets produce subtly-wrong
   text, and exact-match catches what "looks right" misses.
2. **`sample_next`** — temperature, top-k, top-p in the standard order. Fixed-seed graded.
3. **`quantize_groupwise`** — symmetric absmax per group; the skeleton under GPTQ/AWQ.
4. **`speculative_accept`** — the accept/reject rule that makes a draft model free: accept
   with min(1, p_t/p_d), resample rejections from max(0, p_t − p_d). Phase 14 closes the
   loop: your distilled student IS a draft model.

## Run it for real

```bash
python scripts/bench_inference.py runs/150m
```

tok/s uncached vs cached vs quantized; perplexity cost of int8/int4; and the payoff of
Phase 3's GQA: the KV-cache-bytes table (n_kv_head 12 -> 3 = 4x smaller cache).

## Readings

PagedAttention (what vLLM does when many sequences share a GPU), Speculative Decoding,
GPTQ. All in the console.
