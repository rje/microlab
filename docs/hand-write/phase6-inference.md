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
   text, and exact-match catches what "looks right" misses. The cache itself is a second
   graded stub — **`StudentKVCache.append`**: write the new K/V at the current position,
   return the full (0..seq_len) views, and advance `seq_len` **only on the last layer** so
   every layer sees the same positions each step (its `__init__` is provided; the shape
   guard rejects a multi-token step after prefill). `generate_cached` is what drives it.
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

## Serve it

The bench proves the stack is fast; serving proves it's *real*. The four hand-writes above
ARE the serving stack — `generate_cached` + `sample_next` back a live endpoint you can hit
from a browser or the eval harness. Nothing new to write; you're wiring what you already
graded to a socket.

**The endpoint.** `POST /api/generate` on the console streams a completion back as chunked
`text/plain` (token deltas, re-decoded each step so byte-level BPE never splits a multi-byte
character across chunks). Body: `{"prompt", "max_new_tokens" (<=512), "temperature",
"top_k", "top_p", "seed"}`. Auth is either the browser session cookie OR an
`Authorization: Bearer <token>` header — the token is minted into `instance/api_token`
(0600) on first boot, for programmatic clients that can't do the login redirect. Errors are
honest JSON: 400 (empty prompt or over-limit), 401 (bad/missing token), 503 (no checkpoint
to load).

**The Playground.** The console's Playground tab is that endpoint with a UI: type a prompt,
turn the temperature / top-k / top-p / seed knobs, and watch tokens stream in from *your*
150M — served by your KV cache, sampled by your sampler. Seeded generation is
reproducible; drop the seed for fresh samples each run.

**Grade the served model with the same harness.** The Phase 0 eval harness that graded the
Ollama baselines has a `microlab_http` backend that points at `/api/generate`. Same suites,
same checks, now scored against the model you trained and serve:

```json
{
  "name": "microlab-http-150m",
  "backend": {
    "type": "microlab_http",
    "host": "http://127.0.0.1:8765",
    "token_file": "instance/api_token",
    "max_new_tokens": 128,
    "temperature": 0.0
  }
}
```

It posts `seed=0` for determinism and reads the bearer token from `token_file` (or an inline
`token`). This closes the loop the curriculum opened in Phase 0: the harness that measured
other people's models now measures yours, over the wire, through the inference stack you
hand-wrote.

**Stretch — GGUF → Ollama.** Serving through Ollama would let the 150M sit next to the
Qwen baselines under one API. It's *feasible*: our `VariantGPT` is the llama architecture
(RoPE + RMSNorm + SwiGLU), which llama.cpp's `llama` arch already supports, so no kernel
work. The *fiddly* parts are a faithful weight mapping to GGUF tensor names and converting
our byte-level BPE to a vocab llama.cpp accepts. Not worth it until there's something worth
serving broadly — revisit after Phase 9's SFT makes the model instructable.

## Readings

PagedAttention (what vLLM does when many sequences share a GPU), Speculative Decoding,
GPTQ. All in the console.

The serving readings extend the four hand-writes toward what production actually ships.
**EAGLE** is speculative decoding grown up: instead of a separate draft *model*, it trains a
lightweight draft *head* on the target model's own hidden features and verifies a *tree* of
candidate continuations in one pass — the speculative path vLLM and SGLang ship today, a
direct sequel to your `speculative_accept`. **Attention sinks** (StreamingLLM) explains a
sharp failure mode of the KV cache: evict the *first* few tokens to bound memory on a long
stream and quality collapses, because attention dumps leftover probability mass onto those
early positions — keep them as "sinks" and streaming stays stable. **Native Sparse
Attention** (NSA) is the direction *past* GQA: rather than sharing KV heads, make attention
itself natively sparse (compressed + selected + sliding branches) and hardware-aligned so
it's trainable end-to-end, not a post-hoc inference hack — the sparse-attention line
DeepSeek productionized for long-context V-series models.
