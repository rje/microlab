# Scale Infrastructure — Design Spec

- **Date:** 2026-07-01
- **Status:** Approved (target: layered climb to ~1B). Building.

## Goal

Enable the owner's original goal — from nothing to a ~1B model competent enough to be
interesting, learning every step — by adding the **scale infrastructure** the toy oracle
track can't provide. Not a new phase: it's the **real-scale tier** of Phases 1, 2, 4.
The oracle/hand-write track stays the "understand every line" layer beneath it.

## Curriculum mapping

| Phase | Toy layer (done, oracle) | Real-scale layer (this work) |
|---|---|---|
| 1 Data & Tokenization | hand-written BPE + in-memory pipeline | fast 32k BPE (HF `tokenizers`) + FineWeb-Edu → sharded `.bin` streaming |
| 2 Tiny GPT Pretraining | our GPT + toy `train()` | production `Trainer` + first real ~150M run |
| 4 Scaling Experiments | scaling-law oracle | compute-optimal 1B config + the capstone run |

## Components

### 1. Fast tokenizer (`src/microlab/tokenizer/`)
- `train_fast_bpe.py`: train a **32k-vocab** byte-level BPE on a FineWeb-Edu sample via
  HuggingFace `tokenizers` (Rust; tokenizes GB in seconds). Saves `tokenizer.json`.
- `FastTokenizer` wrapper exposing `encode(str)->list[int]` / `decode(list[int])->str`,
  matching the reference BPE interface. **Verified against the reference BPE by round-trip
  + compression** (not merge-equality — different training, both valid). 32k is right-sized
  for 150M–1B (fewer embedding params than GPT-2's 50k).
- Ties to Phase 1: the fast production version of the byte-level BPE hand-written in the
  Phase-1 exercise.

### 2. Data pipeline (`src/microlab/data/prepare.py` + `shard_dataset.py`)
- Stream FineWeb-Edu (HF `datasets` streaming), tokenize with the fast tokenizer, write
  **uint16 `.bin` shards** (~100M tokens each) + a JSON manifest (shard list, token counts,
  tokenizer hash), with a train/val split.
- **Contamination strip**: drop documents containing any Phase-0 eval-suite prompt.
- `ShardDataset`: memmaps the shards and yields `(x, y)` blocks to the GPU (pinned,
  non-blocking) — the scale replacement for the toy `get_batch`.

### 3. Production Trainer (`src/microlab/train/`)
- `Trainer` with: bf16 autocast, AdamW (fused if available), **cosine LR + linear warmup**,
  gradient clipping + accumulation, optional gradient checkpointing, **checkpoint
  save/resume** (model + optimizer + step + RNG + config), periodic **val-loss eval + text
  sample + Phase-0 eval-harness hook**, and VRAM/tokens-per-sec logging.
- Config is a dataclass (`ModelConfig` + `TrainConfig` + `DataConfig`) so **150M and 1B are
  just config files**; the model is `VariantGPT` (RoPE + RMSNorm + SwiGLU from Phase 3)
  scaled up.
- **Resume-equivalence** is the key test: resuming from a checkpoint reproduces the
  uninterrupted trajectory (same loss within fp tolerance).

### 4. Configs + scripts (`configs/`, `scripts/`)
- `configs/150m.py`, `configs/1b.py` (n_layer/n_embd/n_head, block_size, LR schedule,
  batch/accum, token budget ≈ 20× params).
- `scripts/prepare_data.py` (build shards), `scripts/pretrain.py` (run the Trainer from a
  config, resumable).
- **The ~150M run is the validation gate** before committing the multi-week 1B run.

## Papers added (production-scale gaps)

- Phase 1: **The FineWeb Datasets** (Penedo 2024) — the corpus + curation method used here.
- Phase 2: **Cramming** (Geiping 2022) — single-GPU-from-scratch training, this exact
  setup; **Mixed Precision Training** (Micikevicius 2017) — the bf16/autocast mechanics.
- Optional Phase 2/4: **GPT-2** (canonical decoder-only from scratch), **Pythia** (from-
  scratch small-model suite + methodology).

## Relationship to the oracle track (honest)

The production Trainer is **not** oracle-differential-tested — there is no closed-form
oracle for a training loop. It is verified the way nanoGPT is: it drives a real model to
low **validation** loss and coherent samples, and resumes cleanly. This is a
**build-and-verify** subsystem, distinct from the hand-write/oracle phases. The reference
GPT/train stay as the understood versions; the Trainer shares the (scaled) model code.

## Capability expectation (honest)

From-scratch ~1B on ~20B tokens ≈ **GPT-2-XL / Pythia-1B class**: coherent English, shallow
world knowledge, follows simple instructions after SFT, basic reasoning after RL. It will
**not** match modern 1–2B models (Qwen2.5, Llama-3.2), which train on trillions of tokens
(100×+ this budget). The value is a real model built from nothing and understood completely.

## Risks

- **Tokenizer swap** changes vocab (32k) — embedding/lm-head sizes differ from the toy 512;
  configs must set `vocab_size=32000`.
- **Multi-week 1B run**: needs robust checkpoint/resume (interruptions are certain) —
  de-risked by proving the whole pipeline at 150M first.
- **Data at scale**: streaming + memmap avoids loading GBs into RAM; contamination strip
  keeps eval honest.

## Self-review

Covers curriculum mapping, the four components with interfaces, the paper additions, the
honest oracle-vs-build-and-verify boundary and capability ceiling. No placeholders. Scoped
to the data/tokenizer/trainer subsystem; the actual multi-week runs are the owner's to launch.
