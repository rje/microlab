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
  Unit 1 generates v1's own samples on both benchmarks; Unit 2 reuses the HumanEval bank
  and the dedicated test-free MBPP bank (below) — never the standard MBPP bank.
- **Docstring examples cover only 76/164 HumanEval prompts and 0/257 MBPP prompts.** So
  extraction-only selection covers <30% of tasks; the input-synthesis probe (Unit 0)
  decides whether no-tests selection generalizes or stays a HumanEval-subset result.
- **(Review finding C1) The MBPP chat prompt CONTAINS the hidden test asserts verbatim**
  (`mbpp_task` builds the instruction from `test_list`). Any bank generated with the
  standard protocol is test-conditioned at generation — unusable for a "no-tests" claim.
  Unit 2's MBPP work therefore uses a dedicated **test-free bank** (description + bare
  function signature only), and its oracle is computed on that same bank. Bonus
  measurement this yields for free: MBPP pass@1 with vs without in-prompt tests =
  the size of test-conditioning itself.
- **(Review finding I2) On HumanEval, 50 of the 76 docstring-example tasks have example
  inputs that appear verbatim in the hidden test suite.** Clustering on docstring inputs
  therefore discriminates exactly where scoring happens and flatters generalization; the
  Unit 2 table MUST break out docstring-input clustering vs synthesized-input clustering,
  never pool them.
- **(Review finding C2) Statistical power lives on MBPP, not HumanEval.** At v2-proxy
  rates, HumanEval's recoverable gap is ~8 tasks — method differences there cannot reach
  significance (McNemar needs ~6 one-directional discordant pairs). HumanEval results are
  reported as descriptive only; the powered comparison runs on the test-free MBPP bank,
  gated by an explicit power falsifier (below).

## Units

### Unit 0 — Input-DISCRIMINATION probe (the design gate; ~45 min GPU)

Can v1 synthesize inputs that *separate* right code from wrong code? (Review finding I1:
"executes without error" is the wrong property — `f([], 0.5)` can run on every candidate
and partition nothing.) For N=100 fixed tasks (50 HumanEval / 50 MBPP), prompt v1 to
produce 3 call-expressions for the target function. **The synthesis prompt contains the
task DESCRIPTION and signature only — never the gold asserts** (on MBPP the standard
instruction embeds them; the probe strips them, else it measures copying).

An input counts **DISCRIMINATING** iff (a) the reference/canonical solution executes it
without error, AND (b) its output differs from at least one known-wrong solution for the
same task (wrong candidates drawn from the existing eval sample banks; where none exists,
a mechanical mutant of the reference — negated comparison or off-by-one — is used and
labeled as such). Metric: fraction of tasks with ≥2 discriminating inputs.

- **Gate (soft-banded — review minor: at n=100 the fraction itself carries ~5pt SE):**
  ≥56% → synthesized inputs power Unit 2 on both benchmarks. 44–56% → ambiguous zone:
  run Unit 2 both with and without synthesized inputs and report both regimes. 24–44% →
  subset-only, reported as such. <24% → synthesis dead at 1.2B (a finding, echoing the
  repair-probe); Unit 2 restricts to descriptive HumanEval docstring-input clustering and
  the harness ships tests-required.

### Unit 1 — With-tests rerank: complete the known win (~2.5 h GPU)

Generate **v1 sampled n=10** (t=0.7, top-k 40, the established sampled protocol) on
HumanEval AND MBPP via `eval_code.py --n 10`; summarize with the merged `eval_rerank.py`.
Deliverables: v1's true pass@1/pass@10 per benchmark and the **delivered-correctness
table** (oracle rerank = any-of-k). This is the harness's guaranteed-win number: the
standard-protocol banks are the *deployment* numbers (in real use, provided tests ARE in
the prompt — conditioning on them is legitimate there).

**Additionally: the test-free MBPP bank** (per finding C1): n=10 samples from
description+signature prompts, same sampling params. This bank exists solely for Unit 2's
clean no-tests science; its own pass@1/pass@10 (vs the standard bank's) quantifies
test-conditioning as a bonus. (~+45 min GPU.)

### Unit 2 — No-tests selection: how much of the gap survives? (the new science; ~1–2 h GPU beyond Unit 1)

All methods select ONE candidate per task, blind to hidden tests; hidden tests score the
selection afterward. **Benchmark roles (per C1/C2): the test-free MBPP bank is the powered,
clean comparison; HumanEval (standard bank, tests never in HumanEval prompts) is
descriptive corroboration only.** The generator-side leakage rule: no method may consume
text the generator was given that embeds gold tests — enforced by the test-free bank, not
just selector typing.

Selection methods compared:
1. **Random / first-sample** — floor (≈ that bank's pass@1).
2. **Text-plurality** — kept as a *floor* only (review finding I4: the literature's
   "majority vote" for code is majority over EXECUTION OUTPUTS, not text; text-plurality
   fragments and proves little — it is labeled a floor, never the comparison baseline).
3. **Behavioral-output majority / equivalence clustering** — execute all candidates on
   shared inputs; cluster by identical output vectors; select from the largest cluster.
   This IS the literature's strong baseline and our method (they differ only in pick rule);
   the **pick rule is ablated**: shortest-member vs random-member of the largest cluster
   (shortest is an Occam tiebreak, not AlphaCode lineage, and can favor degenerate code —
   measure it, don't assume it). Input sources are **reported separately, never pooled**
   (finding I2): (a) synthesized (truly unseen), (b) docstring-extracted (overlaps hidden
   suite on 50/76 HumanEval tasks — flatters generalization).
4. **Self-test filtering (CodeT-lite)** — v1 generates assert-style checks from the
   test-free prompt; candidates ranked by asserts passed with dual-agreement weighting.
   Gated on Unit 0 (asserts = inputs + expected outputs, strictly harder); recorded as
   gated-out if skipped.
5. **Oracle rerank (hidden tests)** — ceiling, computed on the SAME bank as each
   comparison (test-free MBPP oracle for the powered track).

Primary metric: selected-candidate pass rate on hidden tests, per bank, with per-method
task coverage. Headline: **fraction of the (oracle − floor) gap recovered without tests,
on the test-free MBPP bank**, subject to the power falsifier. Error analysis: when
clustering fails, is the largest cluster a *wrong consensus* (all-agree-wrong — the v2
error-mode data predicts this is common: systematic wrong logic clusters) or
fragmentation? This diagnostic decides whether more samples would help — and
wrong-consensus dominating is a LIVE base-case outcome, not an edge case.

### Unit 3 — Minimal serve integration (engineering; no experiment)

Two explicitly separate deliverables (review minor — don't half-build the gated one):

**3a — tests-required path (plannable NOW, productizes Unit 1's guaranteed win):**
`--best-of k` in the serving/eval path: sample k, execute against caller-provided tests
(an optional asserts field in the request), return the passer. CLI/API only.

**3b — no-tests path (gated on Unit 2's powered result):** return the best cluster member
when no tests are given, and the Playground toggle. Not planned, not scaffolded, until
Unit 2 reports; if Unit 2 lands in its low band, 3b is cancelled and the toggle ships
tests-required. No streaming redesign either way — best-of-n responses render when
selection completes.

## Architecture

Pure, unit-testable selection core in `src/microlab/infer/selection.py`:
`behavioral_signature(outputs: list[str]) -> hashable`, `equivalence_clusters(candidates,
signatures) -> clusters`, `select_by_cluster(clusters) -> candidate`,
`select_by_self_tests(candidates, assert_results) -> candidate`, majority/normalize
helpers — all consuming already-computed execution results (no sandbox calls inside the
pure core). Thin runners: `scripts/probe_input_discrimination.py`, `scripts/eval_selection.py`
(reads Unit-1 sample banks + runs sandbox executions + emits the comparison table),
`--best-of` wiring in serve. Reuses `sample_solutions`, `extract_solution`, `run_python`,
`eval_rerank.delivered`.

## Pre-registered bands (to be committed in the prediction doc before execution)

Indicative shapes the prediction doc will pin exactly — with noise bands sized per the
lab's noise-band discipline (finding I3: the earlier ±2pt band was <1σ of the v1-vs-v2
difference, whose SE is 1.87·√2 ≈ 2.6 pts at n=164):

- v1 true HumanEval pass@10 within **±4 pts (≈1.5σ of the difference)** of v2's 6.1%;
  MBPP pass@10 = 2–3× **the same run's** measured pass@1 (within-run multiplier, no
  cross-model ambiguity).
- Test-free MBPP bank: pass@1 BELOW the standard bank's (test-conditioning is real);
  the drop itself is reported, predicted 20–60% relative.
- Clustering (synthesized inputs, test-free MBPP): recovers **10–50%** of that bank's
  oracle−floor gap — wide and honest: the mechanism analysis (1–2 correct samples in 10;
  systematic wrong-consensus clusters per the v2 error modes) makes LOW recovery a live
  base case, not a tail. Text-plurality floor: <10%.
- Probe discrimination rate: genuinely unknown (the wildcard); the repair-probe precedent
  argues low.

Falsifiers:
- **Power**: recoverable gap (oracle − floor, in tasks) on the powered bank < 15 → declare
  the method comparison UNDERPOWERED; report counts, no recovery-% headline.
- **Leakage (selector-side)**: any method beating its bank's oracle = bug, stop.
- **Leakage (generator-side)**: on any bank, a no-tests method matching oracle specifically
  on tasks whose prompts embedded gold asserts = contamination, not skill; exclude and
  re-report.
- **Probe false-positive**: Unit 0 passes its gate but Unit 2's synthesized inputs yield
  singleton clusters everywhere (no separation in practice) = the probe's discrimination
  metric failed; halt Unit 2, record.
- **Clustering ≤ text-plurality** on the powered bank = execution signal adds nothing at
  this scale; harness ships tests-required only (a finding, not a failure).
- **All-agree-wrong dominates** cluster errors = more samples won't help; caps future
  scaling of k.

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
