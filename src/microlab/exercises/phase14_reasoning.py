"""Hand-write exercise (Phase 14): STaR trace filtering and knowledge-distillation loss.

Fill in the ``NotImplementedError`` bodies so ``tests/exercises/test_phase14_reasoning.py`` passes.
Graded against ``microlab.model.reference.reasoning``. See docs/hand-write/phase14-reasoning.md.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def filter_correct_traces(
    traces: list[str], gold: str, extract_fn: Callable[[str], str | None]
) -> list[str]:
    """STaR filter: keep only traces whose extracted answer matches `gold` (drop ones that
    fail to parse). These become the fine-tuning set — the model learns from its own
    successful reasoning. See docs/hand-write/phase14-reasoning.md."""
    raise NotImplementedError(
        "keep the traces whose extracted answer matches gold; a None extraction never matches"
    )


def distillation_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0
) -> torch.Tensor:
    """Knowledge distillation (Hinton et al.): pull the student's distribution toward a
    frozen teacher's by matching temperature-softened softmaxes. Two design points ARE the
    lesson — which direction the KL runs (which side is the fixed target vs the log-prob
    argument ``F.kl_div`` expects) and the temperature correction that keeps the gradient
    magnitude comparable across temperatures. Zero when student == teacher; returns a scalar.
    See docs/hand-write/phase14-reasoning.md."""
    raise NotImplementedError(
        "soften both logits by the temperature, take the KL between them (mind the direction "
        "and F.kl_div's log-prob/prob argument order), and correct for the temperature so the "
        "gradient magnitude doesn't shrink as T grows"
    )
