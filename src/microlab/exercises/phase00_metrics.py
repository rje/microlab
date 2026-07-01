"""Phase 0 aggregate metrics — YOUR hand-written implementations.

These two functions are the conceptually dense bits of the eval harness — the
lessons from HumanEval/Codex (pass@k) and MMLU (calibration). Everything else in
`microlab.evals` (schema, checks, backends, runner, report, cli) is already built
and green; these two are yours to implement.

Check your work as you go:

    /home/rje/anaconda3/bin/conda run -n microlab pytest tests/evals/test_metrics.py -v

The tests are the spec — make them all pass. See
`docs/hand-write/phase0-pass-at-k-ece.md` for the math, why the naive versions are
wrong, and how these plug into the harness.
"""

from __future__ import annotations

from collections.abc import Sequence


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021, HumanEval / Codex).

    You sampled ``n`` candidate solutions to a problem and ``c`` of them pass the
    unit tests. Estimate the probability that at least one of ``k`` randomly drawn
    samples passes::

        pass@k = 1 - C(n - c, k) / C(n, k)

    Rules the tests enforce:
    - Require ``k <= n``; raise ``ValueError`` otherwise.
    - If there are fewer than ``k`` failures (``n - c < k``), every k-subset must
      contain a passing sample, so return ``1.0``.
    - Return a float in ``[0, 1]``.

    NOTE: the *naive* estimator ``1 - (1 - c/n) ** k`` is biased. Part of the
    exercise is seeing why (the docs explain). ``math.comb`` is your friend.
    """
    raise NotImplementedError("implement pass_at_k — see the docstring and tests")


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (binned).

    A model is *calibrated* when, among predictions it makes with confidence p, a
    fraction ~p are actually correct. ECE measures the gap.

    Partition ``[0, 1]`` into ``n_bins`` equal-width bins. Lower edge inclusive,
    upper edge exclusive, EXCEPT the final bin also includes ``1.0``. For each
    non-empty bin compute ``accuracy`` (fraction correct) and ``confidence`` (mean
    predicted confidence), then::

        ECE = sum over bins of (bin_size / N) * |accuracy - confidence|

    ``confidences[i]`` is the model's confidence in its prediction for example
    ``i`` (a probability in ``[0, 1]``); ``correct[i]`` is whether that prediction
    was right. Empty bins contribute 0. Return a float in ``[0, 1]``.
    """
    raise NotImplementedError(
        "implement expected_calibration_error — see the docstring and tests"
    )
