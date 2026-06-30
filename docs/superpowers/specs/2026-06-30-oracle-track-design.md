# Oracle / Reference Track (Phases 0–2) — Design Spec

- **Date:** 2026-06-30
- **Status:** Approved; building phase by phase.

## Goal

Set up a **reference (oracle) implementation** for each hand-write unit across the
first phases, so the owner can (a) check their own hand-written work against a
known-correct version, and (b) trust that each phase has a tested, stable base the
next phase builds on. The owner still does the work; the reference is the grader and
the foundation. **GPU is first-class** — the owner wants to deal with GPU
idiosyncrasies as part of the learning.

## Structure: reference vs. your work

- **`main`** holds the reference implementations at `src/microlab/<phase>/reference/`.
  They are correct, tested, green, and CUDA-capable — they ARE the stable base. Later
  phases import the reference (Phase 2's GPT imports Phase 1's tokenizer reference);
  building Phase N on Phase N-1's reference is how we validate the base.
- **`learn/<phase>-*` branches** (one per exercise, off main) add the owner's **stub**
  (`src/microlab/<phase>/<module>.py`, raising `NotImplementedError`), the **tests**,
  and a **START-HERE doc**. The owner implements the stub; the `reference/` subpackage
  sits beside it to diff against once attempted.

## The oracle (how tests validate)

Each hand-write unit's tests do up to three things:
1. **Differential** — the owner's stub must match `reference/` exactly given the same
   inputs (and, for nn modules, the same weights via `load_state_dict`).
2. **Property / invariant** — e.g. attention rows sum to 1, the causal mask blocks
   future positions, BPE `decode(encode(x)) == x`, ECE ∈ [0,1].
3. **Overfit-a-batch** (training code) — loss → ~0 on one tiny batch ⇒ the loop is
   correct. Run both on CPU (deterministic) and on the GPU (smoke).

## GPU as first-class

- Reference code is device/dtype explicit; the training loop uses **bf16 autocast**
  (RTX 6000 Ada supports bf16), logs peak VRAM (`torch.cuda.max_memory_allocated`) and
  tokens/sec (with `cuda.synchronize` for honest timing), and supports gradient
  accumulation.
- Hand-write exercises surface GPU concerns: device placement, dtype/precision,
  autocast, deterministic seeding on GPU, OOM/batch-size. START-HERE docs name the
  real gotchas (GPU nondeterminism, dtype mismatches, OOM, timing).
- **Test markers:** GPU/slow tests are `@pytest.mark.gpu`. The pre-commit hook runs
  `pytest -m "not gpu"` (fast); `scripts/check.sh` runs everything (GPU included) as
  the push gate. `gpu` tests `skipif` no CUDA so the suite still passes on a CPU box.

## Per-phase content

### Phase 0 finish
- **Reference:** `pass_at_k`, `expected_calibration_error` (oracle for the existing
  stubs on `learn/phase0-metrics`).
- **Integration that makes them real:** a **pass@k sampling mode** (sample n
  completions per task, score each, aggregate) and a **confidence-emitting backend +
  calibration eval** (collect (confidence, correct), compute ECE) wired into the
  harness. Reference versions on main; the owner's stubs stay on the learn branch.

### Phase 1 — data + tokenizer
- **Reference:** a **source-agnostic** corpus pipeline (raw text → clean →
  exact-dedup → train/val/test split → tokenize) that ships a **tiny bundled
  public-domain sample** (a few KB) so tests run offline/deterministically, plus a
  **BPE tokenizer** (train merges / encode / decode) and a tiny GPU-tensor
  dataset/loader for Phase 2.
- **Data sourcing (a 3-rung ladder, license-clean):** loader recipes for
  (1) **TinyShakespeare** (~1 MB, public domain) for bring-up/first-run;
  (2) a curated **Project Gutenberg** subset (public domain — best for learning the
  pipeline) or **WikiText-103** (CC-BY-SA, known baselines) as the real corpus; and
  (3) **TinyStories** (permissive, HF) for pretraining where a 10–30M model is
  actually fluent. Default: bring up on TinyShakespeare, learn the pipeline on
  Gutenberg, pretrain on TinyStories. Sourced via the HF `datasets` library.
- **Contamination:** the dedup/contamination step strips any overlap between the
  training corpus and the Phase-0 eval suites (the curriculum's contamination check).
- **Hand-write (Tier-1):** the BPE tokenizer (merge loop, encode, decode). The
  pipeline plumbing is review-only.

### Phase 2 — tiny GPT
- **Reference:** a nanoGPT-style decoder-only transformer (token+pos embeddings,
  causal multi-head self-attention, MLP, LayerNorm, blocks, LM head), a CUDA training
  loop (AdamW, cross-entropy, bf16 autocast, VRAM/throughput logging, grad accumulation),
  and a sampler (greedy/temperature/top-k).
- **Hand-write (Tier-1):** scaled-dot-product **attention** (+ causal mask), the
  **transformer block forward**, the **training-step core** (forward→loss→backward→step,
  incl. device/dtype/autocast), and the **sampling loop**. The optimizer is the library.
- **Validation:** differential vs reference (same weights → same logits) and vs
  `F.scaled_dot_product_attention`; property (causal mask); **overfit-a-batch on CPU
  and GPU**; plus a short real GPU training run (script) that drives loss down and
  samples text — proving the base is stable on hardware.

## Execution

Phase by phase, each merged green before the next: **Phase 0 finish → Phase 1 →
Phase 2.** Each phase = reference (on main) + a `learn/<phase>` branch (stubs + tests +
doc). Reported at each phase boundary.

## Scaling to the agent phases (forward-looking)

The **structure** (reference/ subpackage, learn-branch stubs, differential tests,
gpu marker) is domain-agnostic and carries forward. The **validation mode** adapts,
because an agent system splits in two:

- **Mechanical/infra code — oracle works unchanged:** the agent loop (tool-call
  parsing, observation formatting, context management, stop conditions), tool
  implementations, RL math (GAE, PPO/GRPO clipped loss, KL penalty — closed-form, so
  differential-vs-reference fits), and programmatic reward functions. This is most of
  what gets hand-written; same reference + differential/property pattern.
- **Agent behavior — outcome-based eval, not exact-match:** no single golden
  trajectory to match. Validate via task **success rate / reward** over a suite (the
  Phase-0 eval harness is the substrate — Phase 0 is deliberately the seed of agent
  eval), plus **property/invariant** tests on the loop (tool budget, malformed-output
  handling, termination), **golden-trajectory regression fixtures** replayed against a
  *mock* environment + deterministic model (guards the harness, not the policy), and
  LLM-as-judge for open-ended quality.

**Honest limit:** an oracle proves the *code* is correct, not that the *agent is
smart* — that's measurement (real eval runs with statistical significance), a research
activity the harness supports but no unit test can stand in for. **No foundation
change needed now**: `backends.py`'s `FixtureBackend` already seeds the deterministic
mock-model approach, and the eval harness is the plug-in point. Agent phases add a
mock-environment convention + outcome scoring; everything mechanical reuses this track.

## Risks

- **GPU nondeterminism** makes exact differential checks flaky on GPU — so differential
  tests run on CPU (deterministic), GPU tests assert *behavior* (loss decreases, no
  OOM, shapes/dtypes), not bitwise equality.
- **Scope is large** (a from-scratch tokenizer + GPT). Decomposed into three
  independently-mergeable phases; each is its own plan + build + verify.
- **Reference quality matters** (it's the grader). Each reference is itself tested and,
  where a library equivalent exists (attention, BPE), cross-checked against it.

## Self-review

- Covers the approved structure (reference/ subpackage, learn branches), scope (P0+P1+P2),
  GPU-first validation, and the per-phase hand-write/reference split. No placeholders.
- Boundary stated once: reference on main = base + oracle; stubs on learn branches = the
  exercise.
