# Microlab — how to learn how LLMs are built

Microlab is a single-GPU lab for going from nothing to a real language model you
understand line by line. Every phase has the same four layers; you climb through them.

## The four layers of every phase

1. **Read** — the seminal papers for the phase (in the console reading workspace, with an
   AI overview, a section guide, and spaced-repetition flashcards). Reading list per phase
   lives in `site/content/phases.json`.
2. **Understand (the oracle)** — a correct, tested **reference implementation** on `main`
   under `src/microlab/<area>/reference/`. It's the known-good version and the base later
   phases build on. You can read it, run it, and see the concept work on the GPU.
3. **Hand-write (the exercise)** — a **stub you implement** in `src/microlab/exercises/`
   (one file per phase, all on `main` — no branch switching) with tests that grade your work
   **differentially against the oracle** (often by copying the reference's weights into your
   module and asserting identical outputs). Green = provably correct, not just plausible.
   The exercise tests are marked `exercise` and deselected from the guardrail, so `main`
   stays green while your stubs are unsolved. Start with `docs/hand-write/<phase>-*.md`.
4. **Run for real (scale)** — for the pretraining phases (1, 2, 4), the production
   infrastructure to actually train a model: a fast tokenizer, a streaming data pipeline,
   and a checkpoint/resume Trainer. Later phases exercise real runs too — Phases 5–6 profile
   and interpret the trained 150M checkpoint (interp report, inference bench), and Phase 7
   scales the Trainer to multi-GPU for the ~1B capstone. This is *build-and-verify* (no
   closed-form oracle for a training loop) — verified by driving a real model to low
   validation loss.

## The phases

| # | Phase | You hand-write (graded vs oracle) | Real-scale |
|---|---|---|---|
| 0 | Evaluation harness | pass@k, ECE | — |
| 1 | Data & tokenization | byte-level BPE | fast 32k BPE + FineWeb-Edu `.bin` pipeline |
| 2 | Tiny GPT pretraining | attention, block, train step, sampling | production Trainer + 150M run |
| 3 | Architecture ablations | RMSNorm, RoPE, SwiGLU, GQA, MoE routing + load-balance loss | — |
| 4 | Scaling experiments | param/FLOP count, scaling-law fit, muP transfer table, Muon Newton–Schulz step | compute-optimal 1B config + Muon-vs-AdamW A/B |
| 5 | Interpretability | logit lens, induction-head score | interp report on the 150M ckpt |
| 6 | Inference engineering | KV-cached generate, sampling zoo, groupwise quant, speculative accept | inference bench + authed streaming Playground (console serves the 150M) |
| 7 | Distributed training | per-GPU memory budget (DP/TP/PP x ZeRO) | grad-ckpt/compile drills + cloud DDP + 1B capstone |
| 8 | Continued pretraining | forgetting metric, replay mix, RoPE position interpolation | (uses scale) |
| 9 | Supervised fine-tuning | prompt loss-masking, masked CE | (uses scale) |
| 10 | Efficient fine-tuning | LoRA adapter + merge, quantizer | (uses scale) |
| 11 | Reward models | Bradley-Terry preference loss | — |
| 12 | Offline preference opt. | sequence log-prob, DPO loss | — |
| 13 | RL on verifiable tasks | verifiable reward, GRPO advantage, PPO clip | — |
| 14 | Reasoning & distillation | STaR trace filter, distillation loss | — |
| 15 | Tool use & agents | tool-call parse/validate, schema validity | — |
| 16 | Final report | — | — |

## Doing a hand-write exercise (all on `main` — no branch switching)

```bash
cat docs/hand-write/phase2-gpt.md              # the START-HERE guide for the phase
$EDITOR src/microlab/exercises/phase02_gpt.py  # implement the stub in place
pytest -m exercise -k phase02                  # grade against the reference oracle
git commit -am "solve phase 2"                 # your solution is tracked
```
Every exercise is a file in `src/microlab/exercises/` (numbered `phase00`…`phase15`). Its
test is marked `exercise` and deselected from the default guardrail, so `main` stays green
while stubs are unsolved. Attempt first — the reference oracle in
`src/microlab/<area>/reference/` is one folder over to diff against once you've tried. Green
means byte-for-byte agreement with the oracle.

## The scale path (the layered climb to ~1B)

The toy oracle work teaches the mechanics; the scale infra runs them for real. Same model
code (`VariantGPT` with RoPE + RMSNorm + SwiGLU), scaled up:

1. **Tokenizer** — train a 32k BPE (`microlab.tokenizer.fast`) on a data sample.
2. **Data** — `scripts/prepare_data.py` streams FineWeb-Edu → uint16 `.bin` shards
   (`microlab.data.prepare` / `ShardDataset`), stripping eval contamination.
3. **Train** — `scripts/pretrain.py` runs `microlab.train.Trainer` from a config
   (`configs/150m.py`, `configs/1b.py`), resumable across interruptions.
4. **Climb** — prove the whole pipeline at ~150M (~a day), then commit to the ~1B capstone
   (~3–4 weeks locally on the RTX 6000, or ~12–14 h on a rented 8x H100 node, venue decided
   by the Phase 7 vendor spike). See
   `docs/superpowers/specs/2026-07-01-scale-infrastructure-design.md`.

**Optimizer note (Muon):** the Trainer's baseline optimizer is AdamW; the lab is adopting
**Muon** (MomentUm Orthogonalized by Newton-Schulz) for the runs after the 1B. The intuition:
AdamW rescales every scalar weight independently, so a weight matrix's update can concentrate
in a few dominant directions — Muon instead treats the update as a *matrix*, keeping plain
momentum and orthogonalizing it so its singular values flatten and the layer learns in many
directions at comparable strength. Exact orthogonalization (SVD) is too expensive; a few
Newton–Schulz iterations (a handful of matmuls, ~20 lines) approximate it cheaply on the GPU.
It only applies to 2D hidden weight matrices, so the standard recipe is hybrid: matrices →
Muon; embeddings, the tied LM head, and norm gains → AdamW. Learning rates do **not** carry
over 1:1 from AdamW — Muon's update RMS is shape-dependent, and the update-RMS matching from
the Moonshot report is what lets it reuse AdamW-tuned values. Claimed payoff: AdamW-equal
loss at roughly half the FLOPs ("Muon is Scalable for LLM Training", arXiv:2502.16982;
"Practical Efficiency of Muon for Pretraining", arXiv:2505.02222). The Newton–Schulz step is
an oracle-graded Phase 4 hand-write; whether Muon actually wins is build-and-verify — a 150M
Muon-vs-AdamW A/B on otherwise identical configs, scored as steps to equal validation loss,
gates its use for the 1B→2B model-growth run (ablation stretch in
`docs/hand-write/phase4-scaling.md`).

**Honest capability note:** a from-scratch ~1B on ~20B tokens is GPT-2-XL / Pythia-1B class
— coherent, instructable after SFT, basic reasoning after RL. It won't match modern 1–2B
models (trained on trillions of tokens); the value is a real model built from nothing and
understood completely.

## What's oracle-graded vs build-and-verify

- **Oracle-graded** (phases 0–15 hand-writes): a closed-form or reference-differential
  answer exists, so tests prove correctness exactly.
- **Build-and-verify** (the scale Trainer, and — later — whether a trained model/agent is
  actually *good*): no oracle; verified by real runs (val loss, samples, task success on the
  eval harness). An oracle proves your *code* is correct; measurement shows whether your
  *model* is good.

## Datasets

See `docs/datasets.md` for the corpora, licenses, and fetch commands.
