# SOTA parity review: the 1B vs contemporary 1B-class releases

Review run 2026-07-25 (after the fact — the process lesson is that this table must exist
*before* a major run commits). Cohort: the three most comparable shipped models. Every
divergence gets an explicit verdict: **CHOSEN** (divergence stands, with a reason) or
**CHANGED** (we retrofit or fix at the next run).

| | **Microlab 1B** | Llama-3.2-1B | Qwen2.5-1.5B | SmolLM2-1.7B |
|---|---|---|---|---|
| params | 0.98B | 1.24B | 1.54B | 1.71B |
| layers x width | 24 x 1792 | 16 x 2048 | 28 x 1536 | 24 x 2048 |
| attention | **MHA 14:14** | GQA 32:8 | GQA 12:2 (+QKV bias) | **MHA 32:32** |
| context (final) | **1024** | 128k | 32k | 2048 -> 8k (extended) |
| RoPE base | 10k | 500k | 1M | 10k |
| vocab | 32,000 | 128,256 | 151,936 | 49,152 |
| norm / act / tied | RMSNorm / SwiGLU / tied | same | same | same |
| training tokens | **21B** (Chinchilla) | ~9T | 18T | 11T |
| provenance | from scratch | pruned+distilled from 8B | from scratch | from scratch |
| data | FineWeb only | mixed + code/math | mixed + code/math | mixed + code/math |

## Verdicts

1. **Attention (MHA) -> CHANGED.** 2-of-3 cohort models ship GQA; our KV cache is 172KB/token
   (8x Llama-3.2's per-token cost), which taxes serving and every future context increase.
   Notably: `variants.py` **already implements GQA** (`n_kv_head`, oracle track, Phase 3) —
   the production `RunConfig` simply never exposed it. Retrofit: Ainslie et al. (2023, in our
   Phase-3 readings) mean-pool-and-uptrain, target **14:2** (7x KV reduction), ~1B tokens
   (~1 day). SmolLM2 shipping MHA shows this isn't disqualifying — but they paid the same tax.
2. **Context (1024) -> CHANGED.** The whole cohort lands >=8k. SmolLM2's own path (train
   short, extend with RoPE scaling afterward) is exactly our planned staged retrofit
   (4k -> 16k), so this is recoverable at modest cost. Requires plumbing RoPE base/scale
   through VariantConfig/RunConfig (same gap-class as n_kv_head).
3. **Vocab (32k) -> CHOSEN for this model, revisit at next pretrain.** Not retrofittable
   without surgery. 32k is Llama-2-era; the cohort runs 49k-152k. At ~1B params a 128k vocab
   would put ~26% of params in embeddings (tied) — defensible to stay small; SmolLM2's 49k is
   the likely sweet spot to evaluate next time (fertility/compression measured on our mix).
4. **Data mix (FineWeb-only) -> CHOSEN for this run, must-fix at next pretrain.** No
   code/math slice is the cohort's biggest recipe divergence and the likely cause of our
   measured arithmetic floor (probe_track: arithmetic never robust) — consistent with the
   tokenizer's digit handling as co-suspect (queued experiment).
5. **Token budget (21B Chinchilla-optimal vs 9-18T over-trained) -> CHOSEN.** Deliberate:
   the cohort over-trains for inference-cheapness; we optimized for training-compute
   learning value. Over-training is a lever we understand and may pull with Muon later.
6. **Provenance (from scratch, no distillation) -> CHOSEN.** House rule (capability must be
   built, not distilled); SmolLM2 proves the from-scratch path at this scale. Llama-3.2's
   prune+distill is noted as the industrial fast-path we deliberately do not take.
7. **QKV bias / QK-norm -> CHOSEN (none).** Matches Llama-3.2 and SmolLM2; Qwen's QKV bias is
   idiosyncratic. QK-norm belongs to the next generation (Qwen3/Gemma-3) — flagged for the
   next-pretrain parity review, not this cohort.
8. **Process root cause:** the oracle/reference track implemented GQA and ingested the
   PI/YaRN/GQA/attention-sinks papers, but the production RunConfig exposed neither
   `n_kv_head` nor RoPE base — reference capabilities silently unreachable from configs.
   Rule going forward: when the reference track gains a capability, the production config
   surface grows with it (or the divergence is written down as CHOSEN).

## Education-plan gaps found (papers present, application missing)

- **Long-context extension**: 3 papers in readings (PI, YaRN, attention sinks), zero
  curriculum exercise. Added to Phase 3 (see curriculum.md): RoPE-scaling knob + passkey
  eval + staged extension of the 1B.
- **GQA application**: implemented as oracle, never applied to a capstone. Added: the
  MHA->GQA uptrain as a Phase 3 real-scale exercise (conversion is itself the GQA paper's
  core contribution).
- **Tokenizer/vocab sizing**: no curriculum treatment of the vocab-size-vs-model-size
  tradeoff (fertility, embedding share, digit handling). Added to Phase 1.
- **Data mixture design**: no treatment of mixing code/math/web and measuring downstream
  effects — our arithmetic findings make this concrete. Added to Phase 4 as a
  next-pretrain exercise.
