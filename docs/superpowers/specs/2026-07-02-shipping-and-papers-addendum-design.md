# Shipping + Papers Addendum: Serve-It Extension & 13-Paper Batch — Design

**Date:** 2026-07-02
**Status:** Approved (user: "both", after the post-expansion gap review)
**Builds on:** `2026-07-02-curriculum-expansion-design.md` (merged as 84cc0a5)

## Problem

Post-expansion review found the syllabus strong on creation but missing the deployment
half of "LLM creation/deployment": the model never becomes a thing someone can talk to.
Separately, 13 papers (2023–2025 canon + the 2025-26 sparse-attention/speculative story)
fill remaining reading gaps. The user explicitly pulled EAGLE back in: speculative
decoding is increasingly the production-performance lever, and EAGLE is what vLLM/SGLang/
TensorRT-LLM actually ship.

## Part A: 13 papers

| Topic | Paper | arXiv | Read in phase(s) |
|---|---|---|---|
| modern-llm-recipes | Tulu 3: Pushing Frontiers in Open Language Model Post-Training | 2411.15124 | 9, 13 |
| modern-llm-recipes | 2 OLMo 2 Furious | 2501.00656 | 8 |
| tokenizers-data | SmolLM2: When Smol Goes Big | 2502.02737 | 1, 8 |
| foundations | Muon is Scalable for LLM Training | 2502.16982 | 4 |
| foundations | Scaling Data-Constrained Language Models | 2305.16264 | 4 |
| architecture | YaRN: Efficient Context Window Extension | 2309.00071 | 8 |
| architecture | Better & Faster LLMs via Multi-token Prediction | 2404.19737 | 2 |
| architecture | Native Sparse Attention (DeepSeek NSA) | 2502.11089 | 6 |
| inference | Efficient Streaming LMs with Attention Sinks | 2309.17453 | 6 |
| inference | EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty | 2401.15077 | 6 |
| interpretability | Sparse Autoencoders Find Highly Interpretable Features | 2309.08600 | 5 |
| evaluation | SWE-bench: Can LMs Resolve Real-World GitHub Issues? | 2310.06770 | 15 |
| evaluation | tau-bench: Tool-Agent-User Interaction Benchmark | 2406.12045 | 15 |

75 papers total after this batch. arXiv IDs must be TITLE-VERIFIED on download; any
404/mismatch is corrected by searching arXiv for the exact title (report corrections).
Each paper gets the standard treatment (PDF, manifest, overview, cards, synopsis) and
the phases.json readingPaperIds updates above (repeats across phases are allowed —
precedent: scaling-laws appears in phases 2 and 4). EAGLE's synopsis covers the
EAGLE-1→2→3 arc and its role as the production speculative path; NSA's synopsis notes
DeepSeek V4 (Apr 2026) productionized the compressed-sparse-attention direction.

## Part B: "Serve it" — Phase 6 extension

The moment the KV cache stops being an exercise and becomes the serving stack. Three
pieces, all run-for-real (no new oracle):

### B1. Checkpoint loader extraction (enabling refactor)

`load_model` is now duplicated in `scripts/interp_report.py` and
`scripts/bench_inference.py`, and serving needs it a third time. Extract to
`src/microlab/model/reference/checkpoint.py`:

```python
def load_variant_from_run(run_dir: Path, device: str = "cpu") -> tuple[VariantGPT, int]
```
(latest `ckpt_*.pt` by step, builds VariantConfig from the pickled RunConfig, returns
(model.eval(), step); raises FileNotFoundError when no checkpoint — throw, don't mask.)
Both scripts refactor to import it.

### B2. Authed streaming generation endpoint + Playground tab

- `src/microlab/console/serve.py`: lazy singleton serving state —
  `get_state()` loads model+tokenizer once (env: `MICROLAB_SERVE_RUN` default
  `runs/150m`, `MICROLAB_SERVE_TOKENIZER` default `data/shards/tinystories/tokenizer.json`,
  `MICROLAB_SERVE_DEVICE` default `cpu`); a module `threading.Lock` serializes
  generations; `stream_generate(state, prompt, max_new_tokens, temperature, top_k,
  top_p, seed) -> Iterator[str]` — KV-cache prefill then per-token `sample_next`,
  yielding TEXT DELTAS (accumulate ids, decode full, yield the suffix — avoids
  splitting multi-byte BPE artifacts). Hard limits: `max_new_tokens <= 512`;
  `len(prompt_ids) + max_new_tokens <= block_size` else ValueError (400 at the route).
- `POST /api/generate` in `console/app.py`: authed via session OR
  `Authorization: Bearer <token>` where the token lives in `instance/api_token`
  (auto-generated 0600 at first app start, like `secret_key`; bearer path is for the
  eval harness). Chunked `text/plain` streaming response with `X-Accel-Buffering: no`
  (nginx must not buffer) + `Cache-Control: no-cache`. 503 with a clear message when
  no checkpoint/tokenizer exists.
- Console **Playground** tab: third nav view (pattern: the existing Training view).
  Prompt textarea; sliders/inputs for temperature (0–2), top-k, top-p, max tokens
  (≤512); Generate/Stop; streaming output via fetch + ReadableStream; live tok/s and
  latency readout (inference observability, lite).

### B3. Eval-harness HTTP backend (full circle)

`MicrolabHTTPBackend(ModelBackend)` in `evals/backends.py`: posts prompt to
`/api/generate` with the bearer token, collects the stream, returns ModelOutput.
Registered in `create_backend` as type `"microlab_http"` (keys: `host`, `token` or
`token_file`, `max_new_tokens`, `temperature`). The same harness that graded the
Ollama baselines in Phase 0 can now grade YOUR served model over HTTP.

### Docs/content

- Phase 6 guide gains a "Serve it" section (endpoint, playground, HTTP backend, and an
  honest GGUF→Ollama export STRETCH note: llama.cpp's llama arch matches VariantGPT's
  RoPE/RMSNorm/SwiGLU so a `gguf-py` exporter is feasible but fiddly — deferred until
  after SFT, revisit then).
- phases.json phase-6 summary + curriculum.md row 6 run-for-real column mention the
  playground; overview.md Phase 6 deliverables gain the served endpoint.

### Explicitly deferred

GGUF/Ollama export (stretch note only); serving the model on GPU (env flag exists;
default CPU so the live training run is never contended); rate limiting beyond the
single-generation lock.

## Safety/ops constraints

- The live 150M training run must be unaffected: serving defaults to CPU; model loads
  lazily (console memory grows ~0.5GB only after first playground use).
- `/api/generate` must never be reachable unauthenticated (session or bearer only).
- Deploy = merge to main + download papers + npm build + restart microlab-site + live
  verification (playground generates real text on microlab.rje.ai; unauth → 302).
