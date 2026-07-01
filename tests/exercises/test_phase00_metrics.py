"""Spec + validation for the hand-written Phase 0 aggregate metrics.

These tests ARE the spec. Implement `microlab.exercises.phase00_metrics` until they all pass.
They include reference cross-checks (combinatorial + Monte-Carlo for pass@k,
hand-computed bins for ECE) so a subtly-wrong implementation is caught.
"""

import math
import random

import pytest

from microlab.exercises.phase00_metrics import expected_calibration_error, pass_at_k

# --------------------------------------------------------------------------- #
# pass@k
# --------------------------------------------------------------------------- #


def test_pass_at_k_none_correct_is_zero():
    assert pass_at_k(n=5, c=0, k=1) == 0.0


def test_pass_at_k_all_correct_is_one():
    assert pass_at_k(n=5, c=5, k=1) == 1.0


def test_pass_at_k_single_sample_is_fraction():
    assert pass_at_k(n=10, c=3, k=1) == pytest.approx(0.3)


def test_pass_at_k_fewer_failures_than_k_is_one():
    # only 8 failures but we draw 9 -> at least one of the 2 correct must appear
    assert pass_at_k(n=10, c=2, k=9) == 1.0


def test_pass_at_k_rejects_k_greater_than_n():
    with pytest.raises(ValueError):
        pass_at_k(n=3, c=1, k=5)


def test_pass_at_k_matches_combinatorial_definition():
    # exhaustive cross-check against the closed form for many (n, c, k)
    for n in range(1, 13):
        for c in range(0, n + 1):
            for k in range(1, n + 1):
                expected = 1.0 - math.comb(n - c, k) / math.comb(n, k)
                assert pass_at_k(n=n, c=c, k=k) == pytest.approx(expected), (n, c, k)


def test_pass_at_k_matches_monte_carlo():
    # independent check: actually draw k-subsets and measure the pass rate
    rng = random.Random(0)
    n, c, k, trials = 12, 4, 3, 40000
    labels = [True] * c + [False] * (n - c)
    hits = sum(1 for _ in range(trials) if any(rng.sample(labels, k)))
    assert pass_at_k(n=n, c=c, k=k) == pytest.approx(hits / trials, abs=0.01)


# --------------------------------------------------------------------------- #
# expected calibration error
# --------------------------------------------------------------------------- #


def test_ece_perfectly_calibrated_is_zero():
    # conf 0.5, exactly half correct -> accuracy == confidence in that bin
    assert expected_calibration_error([0.5, 0.5, 0.5, 0.5], [True, False, True, False]) == (
        pytest.approx(0.0)
    )


def test_ece_confident_and_all_correct_is_zero():
    assert expected_calibration_error([1.0, 1.0], [True, True]) == pytest.approx(0.0)


def test_ece_hand_computed_two_bins():
    conf = [0.2, 0.4, 0.6, 0.8]
    correct = [False, False, True, True]
    # n_bins=2:
    #   [0.0,0.5): conf {0.2,0.4} acc 0.0 mean-conf 0.3 -> (2/4)*|0-0.3| = 0.15
    #   [0.5,1.0]: conf {0.6,0.8} acc 1.0 mean-conf 0.7 -> (2/4)*|1-0.7| = 0.15
    assert expected_calibration_error(conf, correct, n_bins=2) == pytest.approx(0.30)


def test_ece_single_bin_overconfident():
    # both land in the top bin: acc 0.5, mean-conf 0.95 -> |0.5-0.95| = 0.45
    assert expected_calibration_error([0.95, 0.95], [True, False]) == pytest.approx(0.45)


def test_ece_is_in_unit_interval():
    rng = random.Random(1)
    conf = [rng.random() for _ in range(200)]
    correct = [rng.random() < p for p in conf]
    assert 0.0 <= expected_calibration_error(conf, correct, n_bins=10) <= 1.0


# --------------------------------------------------------------------------- #
# differential vs the reference oracle
#
# The reference lives at microlab.evals.reference.metrics — the known-correct
# version. These randomized checks assert YOUR implementation matches it exactly.
# (Try the exercise before reading reference/metrics.py.)
# --------------------------------------------------------------------------- #


def test_pass_at_k_matches_reference_oracle():
    from microlab.evals.reference.metrics import pass_at_k as ref_pass_at_k

    rng = random.Random(7)
    for _ in range(300):
        n = rng.randint(1, 30)
        c = rng.randint(0, n)
        k = rng.randint(1, n)
        assert pass_at_k(n=n, c=c, k=k) == pytest.approx(ref_pass_at_k(n, c, k)), (n, c, k)


def test_ece_matches_reference_oracle():
    from microlab.evals.reference.metrics import (
        expected_calibration_error as ref_ece,
    )

    rng = random.Random(11)
    for _ in range(50):
        size = rng.randint(1, 60)
        conf = [rng.random() for _ in range(size)]
        correct = [rng.random() < 0.5 for _ in range(size)]
        n_bins = rng.choice([1, 2, 5, 10, 15])
        assert expected_calibration_error(conf, correct, n_bins=n_bins) == pytest.approx(
            ref_ece(conf, correct, n_bins=n_bins)
        ), (size, n_bins)

pytestmark = pytest.mark.exercise
