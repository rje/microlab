"""Spec + validation for the hand-written Phase-11 STaR / distillation primitives.

Implement ``microlab.exercises.phase11_reasoning`` until these pass. Differential tests grade you
against ``microlab.model.reference.reasoning``.
"""

import re

import pytest
import torch

from microlab.exercises.phase11_reasoning import distillation_loss, filter_correct_traces
from microlab.model.reference.reasoning import distillation_loss as ref_distill
from microlab.model.reference.reasoning import filter_correct_traces as ref_filter


def _last_int(s: str) -> str | None:
    m = re.findall(r"-?\d+", s)
    return m[-1] if m else None


def test_filter_correct_traces_known_value():
    traces = ["think... 42", "wrong... 41", "yes 42"]
    assert filter_correct_traces(traces, "42", _last_int) == ["think... 42", "yes 42"]


def test_filter_correct_traces_matches_reference():
    traces = ["a 1", "b 2", "c 1", "d"]
    for gold in ["1", "2", "3"]:
        assert filter_correct_traces(traces, gold, _last_int) == ref_filter(traces, gold, _last_int)


def test_distillation_zero_when_equal():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 16)
    assert distillation_loss(logits, logits).item() == pytest.approx(0.0, abs=1e-6)


def test_distillation_matches_reference():
    torch.manual_seed(1)
    student = torch.randn(3, 4, 10)
    teacher = torch.randn(3, 4, 10)
    for t in (1.0, 2.0, 4.0):
        got = distillation_loss(student, teacher, temperature=t)
        want = ref_distill(student, teacher, temperature=t)
        assert got.item() == pytest.approx(want.item(), abs=1e-5)

pytestmark = pytest.mark.exercise
