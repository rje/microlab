"""Hand-write exercise (Phase 4): the closed-form scaling tools.

Fill in the three ``NotImplementedError`` bodies so ``tests/model/test_student_scaling.py``
passes. The elegant check: your ``count_params`` is graded against the REAL model's
``num_params()`` — derive the formula from the architecture, then the model itself tells
you if you're right. See docs/hand-write/phase4-scaling.md.
"""

from __future__ import annotations

from microlab.model.reference.gpt import GPTConfig


def count_params(config: GPTConfig) -> int:
    """Total parameter count for the reference GPT, computed from the config alone (no
    instantiating the model). Must equal ``GPT(config).num_params()``.

    Account for: tied token embedding (V*C, counted once) + positional embedding (block*C);
    per block — c_attn (3C x C), attn c_proj (C x C), c_fc (4C x C), mlp c_proj (4C x C),
    two LayerNorms (weight+bias each); a final LayerNorm. Linear biases exist only when
    ``config.bias`` is True; LayerNorms always have weight+bias.
    """
    raise NotImplementedError("derive the closed-form parameter count from the config")


def training_flops_per_token(config: GPTConfig) -> int:
    """Approx training FLOPs per token: 6 * (non-embedding parameter count) — the
    Kaplan/Chinchilla ~2N-forward + ~4N-backward rule."""
    raise NotImplementedError("implement the 6N FLOPs estimate (non-embedding params)")


def fit_scaling_law(params: list[int], losses: list[float]) -> tuple[float, float]:
    """Fit L = A * N^(-alpha) and return (A, alpha). Hint: take logs — log L is linear in
    log N with slope -alpha and intercept log A — then least-squares fit the line."""
    raise NotImplementedError("fit the power law via log-log least squares")
