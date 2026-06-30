"""Reference (oracle) implementations of the Phase-0 hand-write metrics.

These are the known-correct versions the owner checks their own `metrics.py`
against. Keep them simple and obviously-correct.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k (Chen et al. 2021): 1 - C(n-c, k) / C(n, k)."""
    if k > n:
        raise ValueError(f"k ({k}) must be <= n ({n})")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> float:
    """Binned ECE. Bins are equal width over [0,1], lower-inclusive/upper-exclusive,
    except the final bin also includes 1.0. ECE = sum_b (|b|/N) * |acc_b - conf_b|."""
    n = len(confidences)
    if n == 0:
        return 0.0
    total = 0.0
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        idxs = [
            i
            for i in range(n)
            if (lo <= confidences[i] < hi) or (b == n_bins - 1 and confidences[i] == 1.0)
        ]
        if not idxs:
            continue
        acc = sum(1 for i in idxs if correct[i]) / len(idxs)
        conf = sum(confidences[i] for i in idxs) / len(idxs)
        total += (len(idxs) / n) * abs(acc - conf)
    return total
