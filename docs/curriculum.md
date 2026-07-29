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
   scales the Trainer to multi-GPU (the ~1B capstone itself trained locally on the RTX 6000
   — see the scale path). Since the 1B landed, the post-training phases (9–13) have real
   runs as well; the table's Real-scale column names them. This is *build-and-verify* (no
   closed-form oracle for a training loop) — verified by driving a real model to low
   validation loss.

## The phases

| # | Phase | You hand-write (graded vs oracle) | Real-scale |
|---|---|---|---|
| 0 | Evaluation harness | pass@k, ECE | lm-eval placement of the 1B vs a public cohort (`docs/benchmarks-1b.md`); likelihood-MC/PMI probe suite; code+tool eval floors (`docs/code-eval-baselines.md`) |
| 1 | Data & tokenization | byte-level BPE | fast 32k BPE + FineWeb-Edu `.bin` pipeline; vocab-sizing/fertility study (run for code: `docs/tokenizer-fertility.md`); licensed + attributed code corpus (`docs/code-corpus-pipeline.md`) |
| 2 | Tiny GPT pretraining | attention, block, train step, sampling | production Trainer + 150M run |
| 3 | Architecture ablations | RMSNorm, RoPE, SwiGLU, GQA, MoE routing + load-balance loss | MHA->GQA uptrain of the 1B (Ainslie mean-pool); staged RoPE context extension (stage 1: 1024->4k shipped, target 16k) + passkey eval |
| 4 | Scaling experiments | param/FLOP count, scaling-law fit, muP transfer table, Muon Newton–Schulz step (stub pending — see Muon note) | compute-optimal 1B config + Muon-vs-AdamW A/B |
| 5 | Interpretability | logit lens, induction-head score | interp report on the 150M ckpt |
| 6 | Inference engineering | KV-cached generate, sampling zoo, groupwise quant, speculative accept | inference bench + authed streaming Playground (console serves your checkpoints, incl. the 1B chat with threaded history + per-model decoding defaults) |
| 7 | Distributed training | per-GPU memory budget (DP/TP/PP x ZeRO) | grad-ckpt/compile drills + cloud DDP + 1B capstone |
| 8 | Continued pretraining | forgetting metric, replay mix, RoPE position interpolation | (uses scale) |
| 9 | Supervised fine-tuning | prompt loss-masking, masked CE | SFT-mix + multi-turn chat SFT of the 1B; chat-aware serving (template, stop strings, threaded history) |
| 10 | Efficient fine-tuning | LoRA adapter + merge, quantizer | (uses scale) |
| 11 | Reward models | Bradley-Terry preference loss | BT reward model on the 1B chat backbone; best-of-n behavioral validation |
| 12 | Offline preference opt. | sequence log-prob, DPO loss | DPO/IPO rounds on the 350M and 1B (UltraFeedback + on-policy RLAIF pairs) |
| 13 | RL on verifiable tasks | verifiable reward, GRPO advantage, PPO clip | GRPO on the 1B against the reward model, KL-leashed, judge-graded |
| 14 | Reasoning & distillation | STaR trace filter, distillation loss | — |
| 15 | Tool use & agents | tool-call parse/validate, schema validity | tool-call eval harness + pre-code-training floors (`docs/code-eval-baselines.md`) |
| 16 | Final report | — | capstone methodology: parity review + design-decision log + verdict audits (see measurement note) |

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
4. **Climb** — prove the whole pipeline at ~150M (~a day), then the ~1B capstone. The 1B
   has now trained locally on the RTX 6000 (21B FineWeb-Edu tokens; the projected 22.6 days
   cut to ~13.5 by TF32 / fused AdamW / max-autotune and the grad-checkpoint headroom work)
   — the Phase 7 cloud drills (DDP scaling, FSDP fit-check on rented GPUs) remain as the
   multi-GPU leg. See
   `docs/superpowers/specs/2026-07-01-scale-infrastructure-design.md`.

**Parity note (config surface):** the 1B capstone shipped MHA at block 1024 even though the
reference track had already implemented GQA (`n_kv_head`) and ingested the PI/YaRN papers —
because the production `RunConfig` never exposed those knobs (see `docs/sota-parity-1b.md`).
Two standing rules: (1) before any major run, table the config against 2-3 contemporary
same-class releases and mark every divergence CHOSEN or CHANGED; (2) when the reference track
gains a capability, the production config surface grows with it. The retrofit exercises
(GQA uptrain, staged context extension with passkey/LAMBADA gates) are now Phase-3 real-scale
items; vocab sizing joins Phase 1 and data-mixture design joins Phase 4 for the next pretrain.

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
"Practical Efficiency of Muon for Pretraining", arXiv:2505.02222). The optimizer itself has
shipped (`microlab.train.muon`, hybrid param groups as above); the Newton–Schulz step is
*slated* as an oracle-graded Phase 4 hand-write but the stub is not yet cut —
`exercises/phase04_scaling.py` ends at the muP table (gap tracked in
`docs/curriculum-audit-2026-07.md`). Whether Muon actually wins is build-and-verify — a
124M twin-arm Muon-vs-AdamW A/B on otherwise identical configs (`configs/muon-ab-*.py`),
scored as steps to equal validation loss, gates its use for the 1B→2B model-growth run
(ablation stretch in `docs/hand-write/phase4-scaling.md`).

**Attention/position succession note:** the Phase-3 attention readings now carry two
successions past what the 1B shipped. On the KV-cache axis: MHA -> GQA (share K/V heads) ->
**MLA** (DeepSeek-V2, arXiv:2405.04434) — instead of sharing heads, compress every head's K/V
into one low-rank latent per token (all heads up-project from a shared `c_KV`, and the
up-projections fold into the Q and O matrices so only the latent is cached: GQA-with-2.25-groups
cache at better-than-MHA quality). Our GQA-conversion audit found exactly the structure MLA
exploits: the 1B's K/V heads look orthogonal in raw weight space but share ~0.42-0.49 of their
structure after basis alignment (`docs/gqa-conversion-audit.md`) — naive mean-pooling ignores
that shared basis, MLA *learns* it as the architecture. MLA is a design candidate for the next
specialist pretrain. On the position axis: RoPE is no longer the settled answer. The NoPE
result (Kazemnejad et al., arXiv:2305.19466) shows decoder-only Transformers length-generalize
*better* with no positional encoding than with RoPE, and the frontier has started acting on
it — Kimi K3 shipped as the first frontier model with globally-NoPE attention (see
[Raschka's K3 architecture notes](https://sebastianraschka.com/blog/2026/kimi-k3-architecture-notes.html)).
Our own RoPE-extension pain is the motivation in miniature: stage 1 of the staged context
extension (1024->4k, `docs/sota-parity-1b.md`) took ABF base retuning, a redone gentler run
after a -2pt short-context regression, and a second anneal to consolidate passkey retrieval —
costs a position-free model would not pay. The NoPE-vs-RoPE ablation has now run at our
scale: 124M twins, identical seeds, 737M tokens — and the length-generalization claim
*inverted* (NoPE collapsed beyond the training window while raw RoPE's loss stayed flat,
and NoPE's passkey retrieval was weak even in-window), so verdict 1 of the design log is **RoPE
stays, pure NoPE rejected** — see `docs/specialist-design-decisions.md` (the live source of
truth) for its verdict-audit status; it reopens only if the hybrid-linear lane wins, where
NoPE's global layers would get position from the recurrent layers, K3-style. The
side finding is a measurement lesson: loss stayed flat where retrieval cliffed to zero, so
loss alone mis-scores position schemes — retrieval evals are mandatory (see the measurement
note). The stretch item is the hybrid linear-attention generation:
Kimi Linear (arXiv:2510.26692) interleaves ~3 linear-attention (KDA) layers per full-attention
layer — and its full layers are MLA running NoPE, so the successions compose — beating matched
full attention while cutting KV cache up to 75%. Hand-write coverage is unchanged (RMSNorm /
RoPE / SwiGLU / GQA / MoE routing are the graded stubs); MLA and NoPE enter through the
readings, the design log's ablation lanes (MLA-vs-GQA is next), and the next-pretrain
parity review.

**Post-training note (the canonical arc, run for real):** Phases 9–13 are no longer paper
exercises — every stage of the classic pipeline has run on the lab's own models, and the
war stories are part of the phase. Phase 9: SFT trains on a *mixed* instruction corpus
(Dolly + Alpaca + No Robots) and chat is **multi-turn** — the chat-mix builder emits whole
conversation paths (OASST) with per-turn loss masking, and serving completes the lesson
(the console Playground threads conversation history through the chat template and stop
strings — Phase 6's endpoint grown up). Phase 11: preference data comes from two sources
worth comparing — public preference sets (UltraFeedback) and **RLAIF** (sample K on-policy
candidates, judge with an external model; house rule: external models may judge, never
supply the capability) — and the Bradley-Terry RM trained on the 1B chat backbone is
validated *behaviorally* by **best-of-n** (RM-argmax of 8 samples beat a single sample 73%
of decided pairs under a position-swapped judge), which is also the cheapest deployment of
an RM. Phase 12: DPO's failure modes were experienced first-hand — over-optimization
degenerates, length bias creeps in — so the readings carry IPO, SimPO, and the
DPO-length-bias paper next to the original. Phase 13: GRPO ran on the 1B against the
learned RM under a **KL penalty** to the frozen SFT reference (KL control is not yet a
graded stub — see the audit), and reward over-optimization is an in-house observation, not
a slide: pushing against the RM inflates its score far faster than judged quality follows,
which is why every RL run is graded by the judge pipeline, never by its own reward curve.

**Measurement & methodology note (the honest-experiment strand):** the lab's recent work
produced as much *method* as model; the curriculum absorbs it as a cross-cutting strand —
each item attaches to an existing phase, not a new one. (1) **Scoring** (Phase 0):
multiple-choice capability is scored by *likelihood* over the answer options, with a
PMI-calibrated variant (subtract each choice's log-prob under a neutral "Answer:" context
to cancel prior frequency), not by parsing free generations; items are written
copy-trap-free with balanced answer positions, and every suite reports its chance baseline
— our old reasoning set scored *below* chance until the copy-traps were removed, which is
the cautionary tale. (2) **Loss is not capability** (Phases 3, 8): in both the 1B context
extension and the NoPE A/B, loss stayed flat while passkey retrieval cliffed — position and
long-context changes must be gated on retrieval evals, never loss alone. (3) **Ablation
discipline** (Phase 4): twin arms, identical seeds and data, models sized to the data,
validation loss; the ladder is a 124M screen → 400–500M when ambiguous → 1B validates the
composition, and no gap counts until it clears the noise band measured by a multi-seed
calibration lane (the Peri-LN lane doubles as ours). (4) **Verdict audits** (Phases 4, 16):
no verdict is final until a positive control reproduces a known-true literature result with
our instrument, an LR sweep rules out tuned-for-the-winner hyperparameters, the gap is
placed against the noise band, and the implementation is re-reviewed against the source
paper; verdicts published before their audit carry PROVISIONAL status
(`docs/specialist-design-decisions.md`). (5) **Reviews at the right vintage** (Phases 4,
16): before committing a design, a parity table against contemporary same-class releases
(the parity note above) and a verified literature sweep at the current year
(`docs/arch-review-2026.md`) — adopt claims at the scale they were demonstrated (this is
how MTP went from a Phase-2 reading to a known skip at our scale). (6) **Benchmark
placement** (Phases 0, 16): compare only within one harness version, machine, and shot
count; write the caveats down (`docs/benchmarks-1b.md` is the worked example). Phase 16 is
where the strand culminates: the capstone is not the model but the audited case for every
claim about it.

**Honest capability note:** a from-scratch ~1B on ~20B tokens is GPT-2-XL / Pythia-1B class
— coherent, instructable after SFT, basic reasoning after RL. It won't match modern 1–2B
models (trained on trillions of tokens); the value is a real model built from nothing and
understood completely. This is now measured, not predicted: under one lm-eval-harness
version on one machine, the 1B leads Pythia-1B (14x our tokens) on 5 of 6 tasks, is roughly
even with GPT-2-XL (1.5x our params), and splits against TinyLlama's 3T-token
over-training; LAMBADA is the one clear loss (training context, data diet, vocab —
`docs/benchmarks-1b.md`).

## What's oracle-graded vs build-and-verify

- **Oracle-graded** (phases 0–15 hand-writes): a closed-form or reference-differential
  answer exists, so tests prove correctness exactly.
- **Build-and-verify** (the scale Trainer, and — later — whether a trained model/agent is
  actually *good*): no oracle; verified by real runs (val loss, samples, task success on the
  eval harness). An oracle proves your *code* is correct; measurement shows whether your
  *model* is good.

## Datasets

See `docs/datasets.md` for the corpora, licenses, and fetch commands.
