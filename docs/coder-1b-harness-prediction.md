# Written BEFORE measurement: what the test-time harness should show

Committed 2026-08-12, before any Unit 0/1/2 measurement runs, per the lab's
pre-registration discipline. Spec: `docs/superpowers/specs/2026-08-11-test-time-harness-
design.md` (independently reviewed; bands below sized per its finding I3 — ≥1.5σ of the
actually-compared quantity, cf. the noise-band rule).

## Baselines in hand (measured previously)

v1 (`runs/coder-1b-instruct-compliant`) greedy: HumanEval 3.0% (5/164), MBPP 14.0%
(36/257). v2-proxy sampled (t=0.7/k40, n=10, HumanEval): pass@1 1.2%, pass@10 6.1% —
**v1's own sampled numbers do not exist yet**; Unit 1 measures them.

## Predictions

**Unit 1 — with-tests (standard banks):**
- v1 HumanEval pass@10 within **±4 pts of 6.1%** (≈1.5σ of the v1-vs-v2 difference, whose
  SE is 1.87·√2 ≈ 2.6 pts at n=164).
- v1 MBPP pass@10 (standard bank) = **2–3× the SAME run's measured pass@1** (within-run
  multiplier; no cross-model ambiguity).
- Oracle delivered-correctness = pass@10 by construction (any-of-k); the delivered table is
  the harness's guaranteed win.

**Test-free MBPP bank (finding C1's clean bank):**
- pass@1 **20–60% relatively BELOW** the standard bank's pass@1 (test-conditioning is real;
  its size is itself a novel measurement). pass@10 drops correspondingly; the bank's own
  oracle−floor recoverable gap is predicted **15–40 tasks** — above the power floor, but
  the power falsifier decides, not this hope.

**Unit 0 — discrimination probe:** genuinely unknown (the wildcard). The repair-probe
precedent (1/221) argues LOW; the task is easier than repair (generate example calls, not
fix logic). Soft-banded gate at 24/44/56% as specced; no directional commitment beyond:
if it lands <24%, that is the second clean "1.2B cannot do X with feedback/spec text"
result, joining self-repair.

**Unit 2 — no-tests selection (powered = test-free MBPP; HumanEval descriptive):**
- Behavioral clustering (synthesized inputs): recovers **10–50%** of the powered bank's
  oracle−floor gap. Wide and honest — the mechanism analysis (1–2 correct samples per
  solved task; systematic wrong-consensus clusters per the v2 error-mode data) makes LOW
  recovery a live base case.
- Text-plurality floor: **<10%** recovery.
- `cluster_random` within noise of `cluster_shortest` (ablation; no directional claim —
  if shortest LOSES clearly, that is the degenerate-code-preference effect, report it).
- Docstring-input clustering (HumanEval, descriptive): recovery ≥ synthesized-input
  recovery on the same tasks (it discriminates where scoring happens — finding I2's
  overlap), and the GAP between them estimates the generalization tax.
- CodeT-lite: gated on Unit 0; if run, between text-plurality and clustering.

## What would falsify it / stop rules (several enforced IN CODE)

- **Selector leak:** any method beating its bank's oracle → `run_methods` RAISES; the run
  aborts. A bug, never a result.
- **Power:** powered-bank recoverable gap < 15 tasks → `summarize` sets `underpowered`;
  report counts ONLY, no recovery-% headline.
- **Generator leak check:** on any bank, a no-tests method matching oracle specifically on
  assert-embedded-prompt tasks → contamination; exclude and re-report.
- **Probe false-positive:** Unit 0 passes its gate but Unit 2 synthesized inputs yield
  singleton clusters nearly everywhere → the probe's metric failed; halt Unit 2, record.
- **Clustering ≤ text-plurality** on the powered bank → execution signal adds nothing at
  1.2B; the harness ships tests-required only (a finding, not a failure).
- **All-agree-wrong dominates** cluster errors → more samples won't help; caps k-scaling.
- **Too good:** clustering recovery > 70% of the gap → check for an input/hidden-test
  overlap path before believing it (the leakage tripwire, house convention).

## Caveats stated up front

- HumanEval numbers are DESCRIPTIVE (recoverable gap ~8 tasks — below any significance
  threshold); no method comparison will be claimed from them.
- The test-free MBPP bank changes TWO things vs the standard bank (no test-conditioning
  at generation AND a different prompt shape); the test-conditioning delta is therefore an
  upper bound on conditioning per se.
- 3b (no-tests serving) ships only if the powered Unit-2 result justifies it; nothing in
  this prediction pre-commits to shipping.
