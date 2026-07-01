"""Hand-write exercise (Phase 11): STaR trace filtering and knowledge-distillation loss.

Fill in the ``NotImplementedError`` bodies so ``tests/model/test_student_reasoning.py`` passes.
Graded against ``microlab.model.reference.reasoning``. See docs/hand-write/phase11-reasoning.md.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def filter_correct_traces(
    traces: list[str], gold: str, extract_fn: Callable[[str], str | None]
) -> list[str]:
    """STaR filter: keep only traces whose extracted answer matches `gold`. These become
    the fine-tuning set (the model learns from its own successful reasoning)."""
    raise NotImplementedError(
        "[t for t in traces if extract_fn(t) is not None and extract_fn(t) == gold]"
    )


def distillation_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0
) -> torch.Tensor:
    """KL(teacher || student) on temperature-softened distributions, scaled by T^2 (so
    gradient magnitude is stable across temperatures). Zero when student == teacher."""
    raise NotImplementedError(
        "s = F.log_softmax(student_logits / temperature, dim=-1); "
        "t = F.softmax(teacher_logits / temperature, dim=-1); "
        "return F.kl_div(s, t, reduction='batchmean') * (temperature * temperature)"
    )
