# Datasets

All corpora are license-clean (public domain or permissive). Downloaded corpora live under
`data/corpora/` (git-ignored — reproducible via the loaders in
`microlab.data.reference.loaders`). Tests never hit the network; they use the tiny bundled
`src/microlab/data/reference/sample.txt` (public-domain Aesop).

## The sourcing ladder

| Rung | Dataset | License | Use | How to get |
|---|---|---|---|---|
| bring-up | **TinyShakespeare** (~1 MB) | public domain | first pretraining run, bring-up | `curl -L -o data/corpora/tinyshakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt` |
| domain shift | **Sherlock Holmes** (~600 KB) | public domain (Gutenberg) | continued-pretraining / forgetting (Phase 8) | `curl -L -o data/corpora/sherlock.txt https://www.gutenberg.org/cache/epub/1661/pg1661.txt` |
| instructions | **Databricks Dolly-15k** | CC-BY-SA 3.0 | SFT (Phase 9) | `curl -L -o data/corpora/dolly15k.jsonl https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl` |
| fluency | **TinyStories** | permissive (HF) | small-model fluency | `load_hf_text("roneneldan/TinyStories")` |
| real pretraining | **FineWeb-Edu** | ODC-BY | the ~150M / ~1B pretraining corpus | HF `datasets` streaming (see `scripts/prepare_data.py`) |
| standard LM | **WikiText-103** | CC-BY-SA | perplexity baselines | `load_hf_text("wikitext", name="wikitext-103-raw-v1")` |

Later phases (11–15) use small task datasets: preference pairs (UltraFeedback / HH-RLHF),
**GSM8K** (~8.5k verifiable math problems, Phase 13), and reasoning traces you generate.

## Which rung for which goal

- **Understand the mechanics** (oracle/hand-write): TinyShakespeare / Sherlock / Dolly are
  plenty — the phenomena (scaling, forgetting, replay, SFT masking, LoRA) all reproduce at
  this scale. See `scripts/validate_oracles.py`.
- **Train a model that actually works**: FineWeb-Edu at real volume (~20 tokens/param).
  This needs the **fast tokenizer** (`microlab.tokenizer.fast`) — the reference BPE is
  O(n·merges) and won't tokenize 10s of MB in reasonable time.

## Regime warning (learned the hard way)

Scaling / ablation experiments must use **models sized to the data** (tokens ≫ params) and
compare **validation** loss. Over-parameterized models on a small corpus overfit, which
inverts the conclusions (bigger looks worse, variants look worse). See
`memory: oracle-validation-regime` and the caveat in `docs/hand-write/phase4-scaling.md`.

## Contamination

The data pipeline (`microlab.data.prepare.strip_contamination`) drops any training document
containing a Phase-0 eval prompt verbatim, so the eval harness stays honest.
