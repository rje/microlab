import math

import pytest

from microlab.evals.reference.calibration import calibration_report, mc_confidence


def test_mc_confidence_argmax_and_softmax():
    pred, conf = mc_confidence({"A": math.log(0.7), "B": math.log(0.2), "C": math.log(0.1)})
    assert pred == "A"
    assert conf == pytest.approx(0.7, abs=1e-6)


def test_mc_confidence_uniform():
    pred, conf = mc_confidence({"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0})
    assert conf == pytest.approx(0.25)


def test_calibration_report_structure_and_ece():
    rep = calibration_report([0.2, 0.4, 0.6, 0.8], [False, False, True, True], n_bins=2)
    assert rep["ece"] == pytest.approx(0.30)
    assert len(rep["bins"]) == 2
    assert all({"lo", "hi", "count", "accuracy", "confidence"} <= set(b) for b in rep["bins"])
