> **Exercise — on `main`, no branch switching.** Implement the stub in `src/microlab/exercises/phase00_metrics.py`, then run `pytest -m exercise -k phase00_metrics` to grade it against the reference oracle. Your solution is tracked in git; commit it when it passes.

# START HERE — hand-write pass@k and ECE (Phase 0)

This is your first hand-written piece. The eval harness around it is **already
built and green** — you implement the two conceptually dense metrics, the lessons
from HumanEval/Codex (pass@k) and MMLU (calibration). Everything else is plumbing
you can read but don't have to write.

You're on the branch the exercises folder on `main`.

## 1. See the harness already works (2 min)

```bash
cd ~/src/python/microlab
./scripts/run_phase0_smoke.sh
# -> total=2 passed=2 failed=0 pass_rate=1.000, writes runs/evals/phase0-smoke-...
```

That ran a tiny suite (`evals/suites/smoke.jsonl`) through the fixture backend and
scored it with the built-in checks (`src/microlab/evals/checks.py`: exact_match,
contains, regex, json_valid, json_field_equals), then wrote a Markdown report.
Skim `src/microlab/evals/` to see the shape: `schema.py`, `suites.py`, `checks.py`,
`backends.py` (fixture + Ollama + HF), `runner.py`, `report.py`, `cli.py`.

## 2. What you implement

Just two functions in `src/microlab/exercises/phase00_metrics.py` (they currently raise
`NotImplementedError`). The tests are the spec:

```bash
/home/rje/anaconda3/bin/conda run -n microlab pytest tests/exercises/test_phase00_metrics.py -v
```

Red until you implement. They include the validation I'd otherwise add by hand: an
exhaustive **combinatorial cross-check** and a **Monte-Carlo cross-check** for pass@k,
**hand-computed bin examples** for ECE, and a **differential check against the
reference oracle**. Green = provably correct, not just plausible.

**The oracle.** A known-correct version now lives at
`src/microlab/evals/reference/metrics.py` (`pass_at_k`, `expected_calibration_error`),
plus `reference/passk.py` (sampling-mode aggregator) and `reference/calibration.py`
(`mc_confidence`, `calibration_report`). The last two tests in `test_metrics.py` diff
your implementation against it on randomized inputs. Try the exercise first — then the
reference is right there to compare against, and to see how pass@k/ECE plug into a real
sampling/calibration eval.

### pass@k — the math

You sampled `n` candidate solutions; `c` pass the unit tests. The chance that `k`
randomly drawn samples contain at least one pass:

```
pass@k = 1 - C(n-c, k) / C(n, k)
```

`C(n-c, k) / C(n, k)` is the probability a random k-subset is drawn entirely from
the `n-c` failures. The naive `1 - (1 - c/n)^k` is **biased** (it assumes sampling
*with* replacement and treats `c/n` as exact) — seeing that bias is half the point.
`math.comb` handles the exact form. Edges: `k > n` → `ValueError`; `n - c < k` →
`1.0`; `c = 0` → `0.0`.

### ECE — the math

Bin predictions by confidence into `n_bins` equal-width bins over `[0,1]` (lower
edge inclusive, upper exclusive, last bin includes 1.0). For each non-empty bin:
`accuracy` = fraction correct, `confidence` = mean confidence. Then:

```
ECE = sum over bins of (bin_count / N) * |accuracy - confidence|
```

A bin that's 95% confident but 50% correct contributes a lot. Watch the bin edge
convention — the hand-computed tests depend on it.

## 3. How they plug into the harness

You don't need to wire these today, but here's where they go so the work feels
real, not abstract:

**pass@k** aggregates over *many samples per task*. Once `pass_at_k` is green:

```python
from microlab.evals.backends import create_backend
from microlab.evals.checks import score_text
from microlab.evals.suites import load_suite
from microlab.evals.metrics import pass_at_k

backend = create_backend({"type": "ollama", "model": "qwen3.6:27b"})
task = load_suite("evals/suites/phase0-core.jsonl")[0]   # a coding task
n = 20
outputs = [backend.generate(task) for _ in range(n)]      # sample n times
c = sum(all(r.passed for r in score_text(task, o)) for o in outputs)
print("pass@1 =", pass_at_k(n, c, 1), " pass@10 =", pass_at_k(n, c, 10))
```

**ECE** aggregates `(confidence, correct)` pairs over a run. It needs the backend to
emit a confidence per prediction (e.g. the max answer-token probability for a
multiple-choice task) — that's a small backend extension we can add next, then:

```python
from microlab.evals.metrics import expected_calibration_error
ece = expected_calibration_error(confidences, correct, n_bins=10)
```

When both are green, the natural next step (my Tier-2 follow-up) is a `pass@k`
sampling mode and a calibration-collecting backend in the runner, plus surfacing the
numbers in the console's Eval Runs panel.

## 4. When you're done

Commit on this branch (the pre-commit hook runs the full suite; for a red
work-in-progress commit use `git commit --no-verify`). When
`pytest tests/exercises/test_phase00_metrics.py` is all green, ping me — I'll do a Socratic
review (I'll ask you *why* `√dₖ`-style questions, not just rubber-stamp) and we
merge to main.

## Optional stretch

- Add a numerically-stable product form of pass@k (the paper's version) and a test
  that it agrees with your `math.comb` version for large `n`.
- Plot a reliability diagram (confidence vs accuracy per bin) — the visual behind
  ECE, useful when you run the real qwen baselines.
