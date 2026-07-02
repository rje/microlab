"""Reference reasoning + distillation tools (Phase 14). STaR bootstraps a reasoning model
by KEEPING only sampled traces whose final answer is correct and fine-tuning on them;
self-consistency takes a majority vote over sampled answers; knowledge distillation trains
a student to match a teacher's softened output distribution."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

import torch
from torch.nn import functional as F


def filter_correct_traces(
    traces: list[str], gold: str, extract_fn: Callable[[str], str | None]
) -> list[str]:
    """STaR filter: keep only traces whose extracted answer matches `gold`. These become
    the fine-tuning set (the model learns from its own successful reasoning)."""
    return [t for t in traces if extract_fn(t) is not None and extract_fn(t) == gold]


def self_consistency(answers: list[str]) -> str | None:
    """Majority vote over sampled answers (the practical test-time reasoning boost).
    Returns the most common answer, or None if the list is empty."""
    if not answers:
        return None
    return Counter(answers).most_common(1)[0][0]


def distillation_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0
) -> torch.Tensor:
    """KL(teacher || student) on temperature-softened distributions, scaled by T^2 (so
    gradient magnitude is stable across temperatures). Zero when student == teacher."""
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (temperature * temperature)
