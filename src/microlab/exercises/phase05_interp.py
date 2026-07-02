"""Hand-write exercise (Phase 5): the two core interpretability primitives — the logit
lens and the induction-head score.

Fill in the ``NotImplementedError`` bodies so ``tests/exercises/test_phase05_interp.py``
passes. Graded against ``microlab.interp.reference.lens``. See
docs/hand-write/phase5-interp.md.
"""

from __future__ import annotations

import torch


def logit_lens(residuals: list[torch.Tensor], ln_f, lm_head) -> torch.Tensor:
    """Decode every layer's residual state through the model's own final norm + unembed.
    residuals: n_layer+1 tensors (B,T,C). Returns (L+1, B, T, V). The final layer's slice
    must equal the model's real output logits."""
    raise NotImplementedError("stack([lm_head(ln_f(r)) for r in residuals])")


def induction_score(attn: torch.Tensor, period: int) -> torch.Tensor:
    """Mean attention mass at offset (i - period + 1) over query positions i >= period.
    attn: (..., T, T) -> (...). A perfect induction head scores 1.0."""
    raise NotImplementedError("gather attn[..., i, i-period+1] for i in [period, T)")
