# Test-time execution harness for coder-1b — design

Written 2026-08-11, after Phase 2 closed (`docs/coder-1b-phase2-milestone.md`). Every
measured arrow points here: SFT saturates in one tranche, RL at 1.2B either damages or
doesn't transfer, self-repair is dead (1/221), and the one quantified reservoir left is the
**pass@k selection gap (~5×)** — harvestable only at inference. Prior art (2024–26:
AlphaCode, Large Language Monkeys, CodeMonkeys, S*, T1, agentic verifiers) all converges on
the same shape: weak per-sample generator + strong external selector.

## Goal

Convert v1's (`runs/coder-1b-instruct-compliant`, **frozen** — no training anywhere in this
project) pass@k reservoir into *delivered* correctness, and measure how much of it survives
when ground-truth tests are NOT available at selection time. Ship best-of-n where
verification exists.

## Global constraints

- **No training.** v1 is frozen; every unit is inference/runtime-side.
- **$0, local only** (RTX 6000 Ada; no renting — standing instruction). Total budget
  ~5–6 GPU-hours + build effort.
- **Executor is the only ground truth.** Hidden benchmark tests are used ONLY for final
  scoring, never for selection in the no-tests experiments (that would be leakage of the
  very thing being measured).
- **Pre-registered prediction** (`docs/coder-1b-harness-prediction.md`) committed before
  any tier-1/2 measurement runs; bands + falsifiers per lab convention.
- No silent fallbacks; every selection method reports how many tasks it could/couldn't
  cover and why (verify by count).

## Data honesty notes (verified, they shape the design)

- **v1 has NO sampled n=10 runs.** The previously quoted "v1 sampled p@1 1.2% / p@10 6.1%"
  was **v2's** measurement used as a proxy (v1≈v2 elsewhere, but it must be measured).
  Tier 1 generates v1's own samples on both benchmarks; every later unit reuses them.
- **Docstring examples cover only 76/164 HumanEval prompts and 0/257 MBPP prompts.** So
  extraction-only selection covers <30% of tasks; the input-synthesis probe (Unit 0)
  decides whether no-tests selection generalizes or stays a HumanEval-subset result.

## Units

### Unit 0 — Input-synthesis probe (the design gate; ~30 min GPU)

Can v1 synthesize *executable inputs* for a problem? For N=60 fixed tasks (30 HumanEval /
30 MBPP), prompt v1 (chat template) to produce 3 call-expressions / stdin lines for the
target function; an input counts VALID if applying it to the task's *reference/canonical
solution* executes without error (correct outputs are NOT required — inputs only need to
run, discrimination comes from executing candidates on them). Metric: fraction of tasks
with ≥2 valid inputs.

- **Gate:** ≥50% → Unit 2 uses synthesized inputs on both benchmarks. 20–50% → Unit 2 runs
  on the synthesizable+example-bearing subset, reported as such. <20% → synthesis is dead
  at 1.2B (recorded as a finding); Unit 2 restricts to the 76 example-bearing HumanEval
  tasks and the majority-vote baseline carries MBPP.

### Unit 1 — With-tests rerank: complete the known win (~2.5 h GPU)

Generate **v1 sampled n=10** (t=0.7, top-k 40, the established sampled protocol) on
HumanEval AND MBPP via `eval_code.py --n 10`; summarize with the merged `eval_rerank.py`.
Deliverables: v1's true pass@1/pass@10 per benchmark and the **delivered-correctness
table** (oracle rerank = any-of-k). This is the harness's guaranteed-win number and the
sample bank all of Unit 2 reuses (no regeneration).

### Unit 2 — No-tests selection: how much of the gap survives? (the new science; ~1–2 h GPU beyond Unit 1)

All methods select ONE candidate from the same k=10 sample bank per task, blind to hidden
tests; hidden tests score the selected candidate afterward.

Selection methods compared:
1. **Random / first-sample** — floor (≈ pass@1).
2. **Majority text-vote** — trivial baseline (exact/normalized-dedup plurality).
3. **Behavioral-equivalence clustering** — execute all candidates on shared inputs
   (synthesized per Unit 0's gate, plus docstring-extracted where available); cluster by
   identical output vectors; pick the shortest member of the largest cluster (AlphaCode /
   symbolic-equivalence-partitioning lineage).
4. **Self-test filtering (CodeT-lite)** — v1 also generates assert-style checks; candidates
   ranked by asserts passed, dual-agreement weighting. Included ONLY if Unit 0 shows
   synthesis works (asserts are inputs + expected outputs — strictly harder); otherwise
   recorded as gated-out, not silently skipped.
5. **Oracle rerank (hidden tests)** — ceiling from Unit 1.

Primary metric per method: selected-candidate pass rate on hidden tests, per benchmark,
with per-method task coverage. Headline: **fraction of the (oracle − floor) gap recovered
without tests.** Error analysis: when clustering fails, is the largest cluster a *wrong*
consensus (all-agree-wrong) or fragmentation? (Decides whether more samples would help.)

### Unit 3 — Minimal serve integration (engineering; no experiment)

`--best-of k` in the serving/eval path: sample k, execute against caller-provided tests
(an optional asserts field in the request), return the passer (or best cluster member when
no tests given and Unit 2 justified clustering). Scope: CLI/API path + a Playground toggle
ONLY if Unit 2's no-tests numbers justify exposing it; otherwise the toggle ships
tests-required. No streaming redesign — best-of-n responses render when selection
completes.

## Architecture

Pure, unit-testable selection core in `src/microlab/infer/selection.py`:
`behavioral_signature(outputs: list[str]) -> hashable`, `equivalence_clusters(candidates,
signatures) -> clusters`, `select_by_cluster(clusters) -> candidate`,
`select_by_self_tests(candidates, assert_results) -> candidate`, majority/normalize
helpers — all consuming already-computed execution results (no sandbox calls inside the
pure core). Thin runners: `scripts/probe_input_synthesis.py`, `scripts/eval_selection.py`
(reads Unit-1 sample banks + runs sandbox executions + emits the comparison table),
`--best-of` wiring in serve. Reuses `sample_solutions`, `extract_solution`, `run_python`,
`eval_rerank.delivered`.

## Pre-registered bands (to be committed in the prediction doc before execution)

Indicative shapes the prediction doc will pin exactly: v1 true pass@10 within ±2 pts of
v2's 6.1% (HumanEval) and 2–3× its pass@1 (MBPP); oracle delivered ≈ pass@10; equivalence
clustering recovers **30–60%** of the oracle−floor gap on covered tasks (prior art:
+10–15 Best@K points at frontier scale — we expect the weak-model fraction, not the
absolute); majority text-vote recovers <15%; input-synthesis validity is the wildcard the
probe measures first. Falsifiers include: clustering ≤ majority-vote (execution signal adds
nothing → harness ships tests-required only); all-agree-wrong dominating cluster errors
(more samples won't help); any method "beating" the oracle (leakage bug — tests reached
selection).

## Testing strategy

Selection core: hermetic unit tests with hand-built execution-result fixtures (clusters,
ties, all-fail, all-agree-wrong). Runners: `--limit` smokes. The leakage falsifier is also
a test: selection functions never receive hidden-test results by construction (type-level
separation of `probe_inputs` vs `hidden_tests` in the runner).

## Risks

- **Input synthesis fails at 1.2B** (repair-probe precedent) → Unit 0 gates the design
  before Unit 2 is built; the fallback path is pre-specified, not improvised.
- **All-agree-wrong consensus** — a weak model can converge on the same wrong algorithm;
  measured explicitly in Unit 2's error analysis.
- **Timeout cost in clustering** — k×inputs sandbox runs with 5s timeouts; capped inputs
  (≤4/task) and short timeouts (≤3s) keep Unit 2 in minutes.
- **Serve-path scope creep** — Unit 3 is deliberately minimal; anything beyond `--best-of`
  + asserts field is out of scope.

## Deliverables

1. `docs/coder-1b-harness-prediction.md` (pre-registered, before tiers run).
2. Unit-0 probe result; Unit-1 v1 sample banks + true pass@k + delivered table;
   Unit-2 selection-accuracy comparison + gap-recovery headline + error analysis.
3. `src/microlab/infer/selection.py` + runners + tests; `--best-of` serve path.
4. Milestone doc scoring every band; updated memory; the ship/no-ship decision for the
   no-tests Playground toggle.
