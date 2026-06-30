"""Reference calibration utilities: turn per-option logprobs into a (prediction,
confidence), and turn collected (confidence, correct) pairs into ECE + a reliability
table (the data behind a reliability diagram)."""

from __future__ import annotations

import math
from collections.abc import Sequence

from microlab.evals.reference.metrics import expected_calibration_error


def mc_confidence(option_logprobs: dict[str, float]) -> tuple[str, float]:
    """Multiple-choice prediction + confidence: softmax over the option logprobs;
    prediction is the argmax option, confidence is its probability (this is how
    MMLU-style scoring yields a confidence)."""
    keys = list(option_logprobs)
    mx = max(option_logprobs.values())
    exps = {k: math.exp(option_logprobs[k] - mx) for k in keys}
    z = sum(exps.values())
    probs = {k: exps[k] / z for k in keys}
    pred = max(probs, key=lambda k: probs[k])
    return pred, probs[pred]


def calibration_report(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> dict:
    n = len(confidences)
    bins = []
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        idxs = [
            i
            for i in range(n)
            if (lo <= confidences[i] < hi) or (b == n_bins - 1 and confidences[i] == 1.0)
        ]
        if idxs:
            bins.append(
                {
                    "lo": lo,
                    "hi": hi,
                    "count": len(idxs),
                    "accuracy": sum(1 for i in idxs if correct[i]) / len(idxs),
                    "confidence": sum(confidences[i] for i in idxs) / len(idxs),
                }
            )
    return {"ece": expected_calibration_error(confidences, correct, n_bins), "bins": bins}
