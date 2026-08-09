# coder-1b-instruct + distill-cost A/B — design

Written 2026-08-09, after the coder-1b base pretrain completed at step 40,000
(`docs/coder-1b-milestone-40000.md`) and the post-train loaders were unblocked for the
hybrid architecture (commit `aa89fed`).

## Goal

Instruction-tune the coder-1b base (`ckpt_40000`, 1.2B, KDA:MLA hybrid, 32k context) into
`coder-1b-instruct` on a fully **compliant** (no-distill) code-instruction mix, and measure
what the "build capability, don't distill" rule costs by training a second, **token-matched
distilled** arm from the same base with identical hyperparameters. Everything runs locally
on the RTX 6000 Ada (49 GB); Phase 1 is **SFT only**.

## Scope boundaries (what this is NOT)

- **SFT only.** Preference optimization (DPO/IPO/GRPO) and the executor-as-reward wiring are
  Phase 2. Adding them here would confound the data-provenance measurement.
- **No self-generation.** The model-proposes / executor-filters loop (SelfCodeAlign-style)
  is Phase 2. At the base's current pass rate (MBPP 10/257) its yield would be thin and
  would conflate "does SFT work" with "does self-gen work." Here the executor only
  **verifies existing human solutions**.
- **No arm C.** The 50/50 A+B blend is deferred; it can be added later at near-zero marginal
  cost once the A/B harness exists.
- **No new training infrastructure.** `sft.py` and the eval harness already exist and are
  unblocked; this project is a data-builder + an experiment harness on top of them.

## Global Constraints

- **Build capability, don't distill.** Arm A contains only human-authored or
  executor-verified-human data. External models may act only as a *judge* (the pairwise
  eval), never as a source of training text. Arm B intentionally violates this — it is a
  disposable measurement instrument (see below), never merged, never used to seed Phase 2.
- **No silent truncation or silent caps.** Overlong examples are dropped and counted, never
  tail-clipped (`collate_sft` right-truncates, which would eat the `### End` stop sentinel).
  Any per-source cap is logged.
- **Identical treatment across arms.** Same base checkpoint, same hyperparameters, same
  decontamination, same eval battery. Only the training data differs.
- **Verify data jobs by count**, not by directory existence — assert per-source row and
  token counts (a pipeline here has twice produced empty output that looked like success).
- Checkpoints and tokenizer follow the existing servable-run convention
  (`ckpt_*.pt` + `tokenizer.json` + `serve_config.json` = chat mode).

## Architecture

Four units with clean interfaces:

1. **Compliant-mix builder** — produces `data/corpora/code_sft_compliant.jsonl`.
2. **Distilled-arm builder** — produces `data/corpora/code_sft_distilled.jsonl`,
   token-matched to arm A.
3. **Decontamination filter** — shared pass applied to *both* mixes against the eval
   benchmarks; emits a report of what was removed.
4. **Experiment harness** — trains each arm via `sft.py`, runs the eval battery, and emits a
   comparison report scored against a pre-registered prediction.

### Unit 1 — Compliant-mix builder (arm A)

Mirrors `scripts/build_sft_mix.py`: pure per-source `normalize_*` functions returning the
`{instruction, context, response}` row (or `{turns: [...]}`), a `build_mix` that
seed-shuffles sources so batches interleave, and a `write_jsonl`. Two modules:

**Module 1 — human sources (build first):**
- **CommitPackFT** (`bigcode/commitpackft`, MIT, ~702k samples, permissive repos only):
  instruction = commit message, response = the new file / diff. Filter to a sensible
  language subset (Python-first, matching the corpus mix), cap per-language to avoid
  message-boilerplate domination.
- **MBPP train split** (`google-research-datasets/mbpp` sanitized; the **train** portion,
  distinct from the test split the eval uses): instruction = problem text, response =
  reference solution.
- **OASST1 code threads** (`OpenAssistant/oasst1`, Apache-2.0): **reuse the tree-walking /
  best-ranked-child logic already in `scripts/build_chat_mix.py`** rather than reinventing
  it; filter to English + code-bearing threads. Provides conversational
  instruction-following style. Emits the multi-turn `{turns: [...]}` schema.

**Module 2 — executor-verified competitive problems:**
- **APPS** (`codeparrot/apps`, MIT), **CodeContests** (`deepmind/code_contests`, data
  CC-BY-4.0), **TACO** (`BAAI/TACO`, Apache-2.0). For each problem, select a human
  submission, assemble it with the problem's tests via `assemble_program`, run it in
  `executor.run_python`, and **keep the pair only if the program exits 0**. Instruction =
  problem statement, response = the verified solution. Dedup TACO against APPS/CodeContests
  (TACO aggregates them). Cap solutions per problem (keep the shortest passing, or a small
  fixed number) so no single problem dominates.

Interface out: one JSONL file in the existing schema. A `--report` flag prints per-source
row + token counts and (for module 2) the pass/fail tally of the verification pass.

### Unit 2 — Distilled-arm builder (arm B)

- **Magicoder-Evol-Instruct-110K** (`ise-uiuc/Magicoder-Evol-Instruct-110K`, GPT-4-authored,
  Apache-2.0 as a file). Normalize to the same schema, then **subsample to match arm A's
  total supervised-token count** (the tokens that contribute to loss, i.e. response +
  sentinel, summed under the coder-1b tokenizer). Matching supervised tokens — not row count
  — is what makes the comparison fair, since arm A's responses (diffs, full solutions) are
  much longer than Magicoder's.
- Emits the same per-source count report.

### Unit 3 — Decontamination filter

A shared function applied to **both** mixes before training. Removes any training example
whose problem/solution overlaps HumanEval, MBPP (test), MBPP+, or LiveCodeBench, by
normalized-substring / n-gram match against the benchmark prompts and canonical solutions.
Emits a report: per-source removals, so a suspiciously high removal count (a sign a source
is benchmark-derived) is visible before spending compute. Decontamination is identical
across arms so it cannot bias the comparison.

### Unit 4 — Experiment harness

- Trains each arm: `sft.py --base-ckpt runs/coder-1b-step40000 --data <mix> --tokenizer
  runs/coder-1b-step40000/tokenizer.json --block-size 2048 --batch-size 2 --grad-accum <to
  hold effective batch> --out runs/coder-1b-instruct-<arm>`. Identical flags except `--data`
  and `--out`.
- Runs the eval battery on each resulting run (below) and writes a comparison report:
  a table of arm A vs arm B across all metrics, plus the pairwise-judge win rate.

## Training configuration

- Base: `runs/coder-1b-step40000/ckpt_40000.pt` (warm start, verified loadable post-`aa89fed`).
- **Block size 2048** (code responses are longer than prose; the measured length
  distribution sets the drop threshold). NoPE + no position params means short-block tuning
  does not disturb the 32k capability — verified by the guardrail eval.
- **Effective batch** anchored to the existing SFT recipe (`sft.py` default batch 16),
  realized as micro-batch × grad-accum. The micro-batch is set empirically: the 16-step
  smoke peaked 45/49 GB at block 1024 / micro-batch 4, so **block 2048 must be validated for
  a safe micro-batch before the full run** (likely micro-batch 1–2 with accum making up the
  effective batch); this is a plan step, not an assumption.
- bf16 autocast, AdamW, cosine LR matched to pretraining, ~3 epochs (tune on arm A first).
- Same seed, schedule, and effective batch for both arms.

## Evaluation + pre-registered prediction

**Pre-registration:** commit `docs/coder-1b-instruct-prediction.md` *before any arm trains*,
with falsifiable bands for each arm on the primary metric and an explicit **distill-gap**
prediction (e.g. "distilled arm beats compliant by X–Y points on HumanEval, or the gap is
within noise"). This is the lab's standing practice and it is what makes the result mean
something rather than being narrated after the fact.

- **Primary:** HumanEval + MBPP pass@1, **greedy and sampled** (`eval_code.py --mode chat`;
  sampled is the honest decoder — greedy loops on this model class and understates it).
- **Secondary:** pairwise judge (`eval_pairwise.py`, external model as judge — within the
  rules) on a held-out set of code instructions not in any training mix.
- **Guardrail:** FIM middle-loss and a passkey long-context check on each instruct arm vs the
  base, to confirm chat-SFT does not damage the base's actual strengths (infill, 32k).

**Verdict rule:** the comparison is only valid if decontamination removed comparable,
explainable amounts from both arms and the guardrail shows neither arm collapsed on FIM /
long-context. A distilled win on a contaminated arm is a false positive and must be caught
by the decontamination report, not the leaderboard.

## Testing strategy

- **Builders:** unit-test the pure `normalize_*` functions (fixture rows → expected schema),
  the token-matching subsampler (arm B total supervised tokens == arm A within tolerance),
  and the executor-verify predicate (a known-passing solution kept, a known-failing one
  dropped) using the existing sandbox tests as a model.
- **Decontamination:** a planted benchmark example is removed; a benign near-miss is kept.
- **Harness:** a tiny end-to-end smoke (limit a few dozen examples, 1 epoch) that trains an
  arm, runs a 2-task eval, and writes the comparison report — proving the wiring, not the
  science.
- Reuse the `_ByteTok` / tiny-config idioms already in `tests/scripts/`.

## Risks being watched

- **APPS/TACO harness variance** — multi-solution, multi-language, differing test formats.
  Mitigation: module 2 is separable and lands after module 1, so a working arm exists even if
  verification is fiddly; cap per-problem and skip-and-count anything that won't assemble.
- **Decontamination leakage** — APPS/TACO overlap the benchmarks. Mitigation: the shared
  filter + visible per-source removal report, mandatory before any paid/long run.
- **FIM/long-context regression** from short-block chat-SFT. Mitigation: the guardrail eval;
  if it regresses materially, revisit block size or add a small infill slice to arm A.
- **Arm-B confound** — provenance *and* style differ simultaneously. Stated honestly: this
  measures the *delivered-capability* cost of the rule, not a mechanistic "is teacher text
  intrinsically better" claim. Token-matching controls volume, not style.

## Deliverables

1. `scripts/build_code_sft.py` (arm A, two modules) + `scripts/build_code_sft_distilled.py`
   (arm B) + a shared decontamination module.
2. `data/corpora/code_sft_compliant.jsonl` and `code_sft_distilled.jsonl` with count reports.
3. `docs/coder-1b-instruct-prediction.md` (pre-registered, committed before training).
4. `runs/coder-1b-instruct-compliant` and `runs/coder-1b-instruct-distilled` checkpoints.
5. An eval comparison + a milestone doc scored against the prediction.
