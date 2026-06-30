import math
import random

import pytest

from microlab.evals.reference.metrics import expected_calibration_error, pass_at_k


def test_pass_at_k_edges():
    assert pass_at_k(5, 0, 1) == 0.0
    assert pass_at_k(5, 5, 1) == 1.0
    assert pass_at_k(10, 3, 1) == pytest.approx(0.3)
    assert pass_at_k(10, 2, 9) == 1.0
    with pytest.raises(ValueError):
        pass_at_k(3, 1, 5)


def test_pass_at_k_combinatorial():
    for n in range(1, 13):
        for c in range(0, n + 1):
            for k in range(1, n + 1):
                expected = 1.0 - math.comb(n - c, k) / math.comb(n, k)
                assert pass_at_k(n, c, k) == pytest.approx(expected), (n, c, k)


def test_pass_at_k_monte_carlo():
    rng = random.Random(0)
    n, c, k, trials = 12, 4, 3, 40000
    labels = [True] * c + [False] * (n - c)
    hits = sum(1 for _ in range(trials) if any(rng.sample(labels, k)))
    assert pass_at_k(n, c, k) == pytest.approx(hits / trials, abs=0.01)


def test_ece_known_values():
    assert expected_calibration_error(
        [0.5, 0.5, 0.5, 0.5], [True, False, True, False]
    ) == pytest.approx(0.0)
    assert expected_calibration_error([1.0, 1.0], [True, True]) == pytest.approx(0.0)
    assert expected_calibration_error(
        [0.2, 0.4, 0.6, 0.8], [False, False, True, True], n_bins=2
    ) == pytest.approx(0.30)
    assert expected_calibration_error([0.95, 0.95], [True, False]) == pytest.approx(0.45)
    assert 0.0 <= expected_calibration_error([0.3, 0.7, 0.9], [False, True, True]) <= 1.0
