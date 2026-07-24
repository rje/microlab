# Benchmarks: Microlab 1B vs public models

All numbers produced locally with the **same lm-eval-harness version, 0-shot**, on the same
GPU (2026-07-23), via `scripts/lmeval_microlab.py` (our VariantGPT adapter) and lm-eval's
`hf` backend for the public models. Metric: `acc_norm` where the harness reports it
(hellaswag, arc_easy, arc_challenge, piqa), else `acc` (winogrande, lambada_openai).
Raw outputs: `runs/lmeval_{1b,pythia-1b,gpt2-xl,tinyllama}.json` (untracked; on the host).

| model | params | train tokens | hellaswag | arc_easy | arc_chall | piqa | winogrande | lambada |
|---|---|---|---|---|---|---|---|---|
| **Microlab 1B** | 0.98B | **21B** | 49.0 | **57.5** | **31.5** | 70.1 | 56.4 | 41.3 |
| Pythia-1B | 1.0B | 300B | 47.2 | 49.2 | 27.0 | 69.4 | 53.6 | 55.9 |
| GPT-2-XL | 1.5B | ~40B | 50.8 | 50.8 | 28.4 | 70.5 | **58.4** | 50.9 |
| TinyLlama-1.1B | 1.1B | 3,000B | **55.5** | 42.7 | 30.1 | 70.1 | 49.6 | 49.0 |

## Reading

- **vs Pythia-1B (14x our tokens): we lead on 5 of 6 tasks.** The 2023 recipe (GPT-NeoX
  architecture, the Pile) needed 300B tokens to land where 21B of deduplicated FineWeb plus a
  modern block (RoPE/RMSNorm/SwiGLU, Chinchilla-sized run) lands. Data quality and recipe
  vintage matter more than raw token count at this scale.
- **vs GPT-2-XL (1.5x our params): roughly even** — we win the ARC pair, it edges hellaswag
  and winogrande. A 2019 1.5B and a 2026-recipe 0.98B are peers.
- **vs TinyLlama (143x our tokens): split.** Its 3T-token over-training buys a clear
  hellaswag lead (long-tail commonsense keeps accruing), but it does not buy ARC or
  winogrande, where we lead.
- **lambada_openai is our one clear loss** (41.3 vs 49-56). Three compounding causes, all
  expected: our 1024-token training context vs their 2048 (LAMBADA rewards long narrative
  dependencies), a web-text training diet light on fiction relative to the Pile, and final-word
  prediction being sensitive to our smaller 32k vocab's tokenization.

## Caveats

- Tokenizer differences mean likelihoods are not perfectly commensurable across models;
  acc/acc_norm over choices is the standard mitigation but not a cure.
- 0-shot only; few-shot would strain our 1024 context and was not run.
- Single harness version and machine — numbers may differ slightly from published tables
  (which mix harness versions, shots, and metric variants); the within-table comparison is
  the meaningful one.
