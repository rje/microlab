# Curriculum Expansion: Inference, Interpretability, Distributed Training — Design

**Date:** 2026-07-02
**Status:** Approved pending review

## Problem

A gap analysis of the 14-phase curriculum found it strong on the training pipeline
(pretraining → SFT → preferences → RL → distillation) but systematically missing
everything after training finishes, plus several concepts that drive modern
architecture decisions:

1. **Inference & serving** — KV cache, sampling strategies, inference quantization,
   speculative decoding, batching. Conceptually load-bearing: GQA exists *because* of
   the KV cache; generation is memory-bound *because* of it.
2. **Modern attention topologies** — GQA/MQA/MLA. Llama 3 and DeepSeek-V3 are in the
   reading list but the curriculum never builds or ablates their attention shapes.
3. **Distributed training** — DP/TP/PP/EP, ZeRO, FSDP. Core to "how labs actually do
   it"; previously excluded by the single-GPU constraint, now unlocked by a cloud
   budget (hundreds of dollars, not thousands).
4. **Scaling-era craft** — muP hyperparameter transfer, training-stability/loss-spike
   literacy. Directly needed to pick the 1B run's learning rate honestly.
5. **MoE hands-on** — Switch Transformers is read but never built.
6. **Long context** — RoPE position interpolation, context extension.
7. **Interpretability** — logit lens, induction heads. The most direct route to *why*
   transformers work, using the from-scratch 150M model we own end to end.
8. **Data recipe modernities** — annealing/midtraining (reading-level addition only).

Deliberate exclusions, now recorded as decisions rather than omissions: multimodality,
RAG/retrieval, deep safety work (red-teaming, jailbreak evaluation).

## Constraints

- **Educational-first**: every addition follows the repo's four-layer pattern
  (Read → Oracle → Hand-write → Run-for-real) from `docs/curriculum.md`.
- **Sequencing**: each phase must build on prior phases; motivation should land
  before mechanism where possible (and be explicitly called back where not).
- **Cloud budget**: hundreds of dollars total is acceptable; thousands is not.
  Estimates below cap at ~$450 worst case including the 1B capstone.
- **1B capstone venue is deferred**: when Phase 7 arrives, a research spike on vendor
  affordability (Lambda, and peers) decides local-vs-cloud. Both paths are designed.
- No student work exists on phases 5+ (no learn branches), so renumbering is safe.

## New phase sequence (17 phases, 0–16)

| # | Phase | Change | Hand-write (graded vs oracle) |
|---|---|---|---|
| 0 | Evaluation harness | unchanged | pass@k, ECE |
| 1 | Data & tokenization | unchanged | byte-level BPE |
| 2 | Tiny GPT pretraining | unchanged | attention, block, train step, sampling |
| 3 | Architecture ablations | **+ GQA, + tiny MoE** | RMSNorm, RoPE, SwiGLU, **GQA attention, top-k router + load-balance loss** |
| 4 | Scaling experiments | **+ muP, + stability reading** | param/FLOP count, scaling-law fit, **muP scaling rules** |
| **5** | **Interpretability** (new) | — | **logit lens, induction-head score** |
| **6** | **Inference engineering** (new) | — | **KV-cached generate, sample_next (temp/top-k/top-p), groupwise int4 quant, speculative accept rule** |
| **7** | **Distributed training** (new) | — | **per-GPU memory budget under DP/TP/PP + ZeRO** |
| 8 | Continued pretraining | **+ long context, + annealing reading** | forgetting metric, replay mix, **RoPE position interpolation** |
| 9 | Supervised fine-tuning | renumbered (was 6) | unchanged |
| 10 | Efficient fine-tuning | renumbered (was 7) | unchanged |
| 11 | Reward models | renumbered (was 8) | unchanged |
| 12 | Offline preference opt. | renumbered (was 9) | unchanged |
| 13 | RL on verifiable tasks | renumbered (was 10) | unchanged |
| 14 | Reasoning & distillation | renumbered (was 11); add speculative-decoding callback (draft model = distilled student) | unchanged |
| 15 | Tool use & agents | renumbered (was 12) | unchanged |
| 16 | Final report | renumbered (was 13) | — |

### Sequencing rationale

- **GQA taught twice on purpose**: Phase 3 ablates it as a quality/throughput tradeoff
  ("n_kv_heads=3 barely hurts loss"); Phase 6 measures its 4× KV-cache shrink — the
  *reason it exists* lands after the student has felt the cache. The Phase 3 guide
  forward-references this.
- **Interp (5) before inference (6)**: the arc is *train it → look inside it → make it
  fast → adapt it*. Both use the trained 150M checkpoint.
- **muP in Phase 4** is used immediately: the Phase 7 capstone's 1B learning rate is
  chosen by transfer from the scaling family, not vibes.
- **Distributed (7) directly precedes what it enables** — the 1B capstone closes the
  phase.
- **Speculative decoding (6)** plants the draft-model concept that distillation (14)
  completes.

## Per-phase design

### Phase 3 additions (Architecture ablations)

**Oracle:** extend `VariantConfig`/`VariantGPT` with `n_kv_head: int | None = None`
(None → n_head, i.e. MHA; existing checkpoints unaffected). Implement grouped-query
attention in `model/reference/variants.py` (KV heads repeated-interleaved to serve
query groups). New `model/reference/moe.py`: `TopKRouter` (softmax over expert logits,
top-k dispatch) + `MoEMLP` (Switch-style, per-expert SwiGLU) + `load_balance_loss`
(Switch aux loss: fraction-dispatched × mean router prob, scaled by n_experts).

**Hand-write (`exercises/phase03_variants.py`, extended):**
- `class GQAttention(nn.Module)` — graded by copying reference weights, asserting
  identical outputs; parametrized over n_kv_head ∈ {1, 3, n_head} (MQA, grouped, MHA).
- `def route_topk(logits, k)` and `def load_balance_loss(router_probs, dispatch_mask)`
  — closed-form diff vs oracle.

**Run for real:** add `n_kv_head` to the ablation matrix; measure val loss + throughput
on TinyShakespeare-scale runs like the existing ablations.

**Readings added:** MQA (arXiv 1911.02150), GQA (arXiv 2305.13245). Switch
Transformers already present.

### Phase 4 additions (Scaling experiments)

**Oracle:** `model/reference/scaling.py` gains `mup_multipliers(base, target)` →
per-group dict: hidden-weight LR × (base_width/width), output-layer init/mult rules,
attention 1/d (vs 1/√d) scaling — the muP table as closed-form code.

**Hand-write:** implement `mup_multipliers` from the paper's Table; graded exactly.

**Run for real:** coordinate check — activation RMS across widths {64, 128, 256} stays
flat under muP scaling and blows up under naive scaling; one plot. LR-transfer
mini-sweep at two small widths showing the optimum lands at the same muP-space LR.

**Readings added:** muP / Tensor Programs V (arXiv 2203.03466), small-scale proxies
for training instabilities (arXiv 2309.14322).

### Phase 5: Interpretability (new)

**Goal:** open up the trained 150M and find real structure — the phase where "I built
it" becomes "I can see what it learned."

**Oracle (`src/microlab/interp/reference/lens.py`):**
- `logit_lens(hidden, ln_f, lm_head)` — project each layer's residual stream through
  final LN + unembedding; returns per-layer logits.
- `induction_score(attn)` — given attention patterns on a repeated random-token
  sequence [A B C … A B C …], score each head on attention mass at position (i − L + 1)
  (the token after the previous occurrence).

**Hand-write (`exercises/phase05_interp.py`):** both functions, diff-graded on fixed
tensors and on real 150M activations.

**Run for real:** `scripts/interp_report.py` against the 150M checkpoint — logit-lens
table for a prompt (watch the prediction sharpen layer by layer), per-head induction
scores, attention-map images for the top induction heads; artifacts into `runs/interp/`.
Stretch (optional): re-run induction scoring on saved intermediate checkpoints
(ckpt_200…N exist thanks to disabled pruning) to catch the induction-head formation
during training.

**Readings:** Tuned Lens (arXiv 2303.08112), ROME (arXiv 2202.05262); Anthropic's
"In-context Learning and Induction Heads" linked in the phase summary (web-only pub,
not in the PDF library).

### Phase 6: Inference engineering (new)

**Goal:** everything between a checkpoint and a served token; why inference is
memory-bound and what the field does about it.

**Oracle (`src/microlab/infer/reference/`):**
- `kv_cache.py` — `KVCache` (per-layer k/v tensors, append + view) and
  `generate_cached(model, idx, n, temperature)`; VariantGPT forward gains an optional
  `kv_cache`/`start_pos` path (RoPE offset by cache length).
- `sampling.py` — `sample_next(logits, temperature, top_k, top_p, generator)`; exact
  reference semantics (temp=0 → argmax; top-p smallest set ≥ p).
- `quant.py` — `quantize_groupwise(w, bits, group_size)` / dequant (absmax per group);
  int4/int8.
- `speculative.py` — `speculative_accept(draft_probs, target_probs, draft_tokens,
  generator)` implementing the Leviathan accept/reject + residual-resample rule.

**Hand-write (`exercises/phase06_inference.py`):** all four. `generate_cached` is
graded by exact token-match against uncached reference generation (greedy) — the
sharpest correctness test in the curriculum — plus a wall-clock speedup assertion.

**Run for real:** `scripts/bench_inference.py` on the 150M: tok/s uncached vs cached
vs cached+int8; perplexity before/after quantization (reuses `evaluate_perplexity`);
GQA-vs-MHA KV-cache bytes table (calls back to Phase 3). Numbers into the phase note.

**Readings:** PagedAttention/vLLM (arXiv 2309.06180), speculative decoding
(arXiv 2211.17192), GPTQ (arXiv 2210.17323).

### Phase 7: Distributed training (new)

**Goal:** the parallelism vocabulary of every frontier lab, felt on real hardware at
least once. Ends with the 1B capstone.

**Oracle (`src/microlab/distributed/reference/memory.py`):**
`memory_budget(cfg, world, dp, tp, pp, zero_stage, dtype)` → per-GPU bytes for
{params, grads, optimizer states, activations} under the given parallelism; the
closed-form bookkeeping behind "will it fit."

**Hand-write (`exercises/phase07_distributed.py`):** `memory_budget`, graded against
the oracle across a matrix of configs (1B/7B/70B × ZeRO 0–3 × dp/tp). Pencil-and-paper
knowledge made executable.

**Run for real (three rungs):**
1. **Local, free:** add `grad_checkpoint: bool` and `compile: bool` to `RunConfig`;
   measure VRAM/throughput deltas on the 150M config.
2. **Cloud drills (~$25–50):** rent 4× A100 on Lambda for an afternoon. Runbook in
   `ops/lambda-distributed.md`. `scripts/pretrain_ddp.py` (torchrun wrapper: DDP model,
   rank-sharded data, all-reduced metrics — reuses Trainer) → measure DDP scaling
   efficiency 1→4 GPUs on the 150M; then FSDP the 1B config and verify the memory
   budget predictions from the oracle against `nvidia-smi` reality.
3. **1B capstone — venue deferred:** first task of the capstone is a **research spike
   on vendor affordability** (Lambda and competitors; spot vs on-demand; $/GPU-hr for
   8× H100 vs 4× H100 vs big single GPUs). Decision inputs documented in the phase
   note. Paths: (a) cloud 8× H100, ~12–14 h, ~$300–400; (b) local RTX 6000, free,
   ~3–4 weeks with grad-checkpointing + compile. Either way the muP-transferred LR and
   the memory-budget oracle get used for real.

**Readings:** Megatron-LM (arXiv 1909.08053), ZeRO (arXiv 1910.02054).

### Phase 8 additions (Continued pretraining, was 5)

**Oracle:** `rope_interpolate(cos, sin, scale)` (or position-scaling variant) in
`model/reference/continued.py`.

**Hand-write:** the interpolation function, diff-graded.

**Run for real:** take the 150M (block 1024), evaluate ppl at 2048/4096 raw (bad),
with position interpolation (better), after a short interpolated finetune (best).
Annealing/midtraining enters as reading + a data-mix note tied to the existing
`build_replay_mix` exercise (Llama 3 report sections; no new oracle).

**Readings added:** Position Interpolation (arXiv 2306.15595).

## Papers added (12)

| Topic (folder) | Paper | arXiv |
|---|---|---|
| architecture | Fast Transformer Decoding: One Write-Head is All You Need (MQA) | 1911.02150 |
| architecture | GQA: Training Generalized Multi-Query Transformer Models | 2305.13245 |
| architecture | Extending Context Window via Positional Interpolation | 2306.15595 |
| foundations | Tensor Programs V: muP hyperparameter transfer | 2203.03466 |
| foundations | Small-scale proxies for training instabilities | 2309.14322 |
| interpretability (new) | Eliciting Latent Predictions with the Tuned Lens | 2303.08112 |
| interpretability (new) | Locating and Editing Factual Associations in GPT (ROME) | 2202.05262 |
| inference (new) | Efficient Memory Management for LLM Serving (PagedAttention) | 2309.06180 |
| inference (new) | Fast Inference via Speculative Decoding | 2211.17192 |
| inference (new) | GPTQ: Accurate Post-Training Quantization | 2210.17323 |
| systems (new) | Megatron-LM: Model Parallelism | 1909.08053 |
| systems (new) | ZeRO: Memory Optimizations Toward Trillion-Parameter Models | 1910.02054 |

Each gets the standard treatment: PDF under `papers/<topic>/`, manifest entry,
synopsis, overview.json + cards.json for the console reading workspace.

## Renumbering (mechanical)

- `exercises/phase05_continued.py → phase08_continued.py` … `phase12_tools.py →
  phase15_tools.py` (+3 each); same for `tests/exercises/test_phase*` and
  `docs/hand-write/phase*` guides; fix docstring cross-references.
- `site/content/phases.json`: ids `phase-5…phase-13` → `phase-8…phase-16`; insert new
  phase-5/6/7 entries; update readingPaperIds.
- `docs/curriculum.md` table and `plans/llm-lab-overview.md` phase plan rewritten;
  overview gains the cloud-budget assumption and the explicit out-of-scope list
  (multimodality, RAG, deep safety).

## Cost summary

| Item | When | Est. |
|---|---|---|
| Phase 7 local rung | Phase 7 | $0 |
| Phase 7 cloud drills (4× A100 afternoon) | Phase 7 | $25–50 |
| 1B capstone | Phase 7 end | $0 (local) or ~$300–400 (cloud) — deferred, vendor spike decides |
| **Worst case total** | | **~$450** |

## Implementation order

1. Renumbering commit (files, tests, guides, phases.json, curriculum docs) — keeps
   main green on its own.
2. Papers: fetch 12 PDFs, manifest entries, new topic folders; synopses/overview/cards
   content.
3. Phase 3 additions (GQA in variants + MoE oracle, stubs, tests, guide update).
4. Phase 4 additions (muP oracle, stub, test, guide update).
5. Phase 5 interp (new area `src/microlab/interp/`, oracle, stubs, tests, guide,
   `scripts/interp_report.py`).
6. Phase 6 inference (new area `src/microlab/infer/`, oracle incl. VariantGPT
   kv-cache forward path, stubs, tests, guide, `scripts/bench_inference.py`).
7. Phase 7 distributed (oracle, stub, test, guide, `RunConfig` grad-ckpt/compile
   flags, `scripts/pretrain_ddp.py`, `ops/lambda-distributed.md`).
8. Phase 8 long-context additions (oracle, stub, test, guide update).
9. Console content pass + deploy + live verification.

Each step lands as its own commit(s) with tests green; exercise tests marked
`exercise` and deselected from the guardrail per repo convention (`main` stays green
with unsolved stubs).

## Testing

- All new oracles get non-exercise unit tests (correctness of the reference itself),
  mirroring how existing references are tested.
- All new stubs get `exercise`-marked differential tests.
- KV-cache reference is additionally tested for exact-match with the existing
  uncached `generate` (greedy) — guards the VariantGPT forward-path change.
- `n_kv_head=None` default asserted to reproduce current VariantGPT outputs
  bit-for-bit (checkpoint compatibility for the live 150M run).
- Site content validated by the existing console content-validation on load;
  post-deploy live check of the new phase pages.
