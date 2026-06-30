"""Reference pass@k sampling mode: sample n completions per task, score each with the
harness checks, count how many pass, and estimate pass@k for several k."""

from __future__ import annotations

from collections.abc import Iterable

from microlab.evals.backends import ModelBackend
from microlab.evals.checks import score_text
from microlab.evals.reference.metrics import pass_at_k
from microlab.evals.schema import EvalTask


def passk_eval(
    backend: ModelBackend, task: EvalTask, n_samples: int, ks: Iterable[int]
) -> dict[int, float]:
    outputs = [backend.generate(task) for _ in range(n_samples)]
    c = sum(1 for o in outputs if all(r.passed for r in score_text(task, o)))
    return {k: pass_at_k(n_samples, c, k) for k in ks}
