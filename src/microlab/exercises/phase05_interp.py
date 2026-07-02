"""Hand-write exercise (Phase 5): the interpretability primitives — residual-stream
capture, the logit lens, attention-pattern extraction, and the induction-head score.

Fill in the ``NotImplementedError`` bodies so ``tests/exercises/test_phase05_interp.py``
passes. Graded against ``microlab.interp.reference.lens``. See
docs/hand-write/phase5-interp.md.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def collect_residual_stream(model, idx: torch.Tensor) -> list[torch.Tensor]:
    """Mirror ``VariantGPT.forward`` up to (not through) the final norm, capturing the
    residual stream after the embedding and after each transformer block. Returns n_layer+1
    tensors of shape (B, T, C), oldest (post-embedding) first. Note the pos variants: the
    'learned' block adds a positional embedding, the RoPE block does not. Graded elementwise
    vs the reference. See docs/hand-write/phase5-interp.md.
    """
    raise NotImplementedError(
        "run the model's own forward by hand, keeping x after the embedding and after every "
        "block — this is the list the logit lens decodes"
    )


@torch.no_grad()
def attention_patterns(model, idx: torch.Tensor) -> torch.Tensor:
    """Recompute the softmax attention probabilities per layer/head for the RoPE block —
    SDPA never materializes them, so you rebuild the pattern by hand. Asserts pos == 'rope'.
    Returns (n_layer, n_head, T, T) for the batch's first sequence. Handle BOTH attention
    variants: plain RoPE attention, and the GQA branch where the k/v projection has fewer
    heads that must be expanded up to n_head before scoring. See
    docs/hand-write/phase5-interp.md.
    """
    raise NotImplementedError(
        "per block: normed input -> q,k via the block's own attn weights -> RoPE -> scaled "
        "scores -> causal mask -> softmax; stack per layer/head (mind the GQA branch)"
    )


def logit_lens(residuals: list[torch.Tensor], ln_f, lm_head) -> torch.Tensor:
    """Decode every layer's residual state through the model's own final norm + unembed —
    what the model would predict if forced to stop there. residuals: n_layer+1 tensors
    (B,T,C). Returns (L+1, B, T, V); the final layer's slice must equal the model's real
    output logits. (nostalgebraist's logit lens; Tuned Lens is the learned-probe upgrade.
    See docs/hand-write/phase5-interp.md.)
    """
    raise NotImplementedError(
        "apply the head's final ops (ln_f then lm_head) to each captured residual, one output "
        "row per layer — see the docstring"
    )


def induction_score(attn: torch.Tensor, period: int) -> torch.Tensor:
    """An induction head, at query i, attends to the token AFTER the previous occurrence of
    the current token — offset (i - period + 1), a single fixed sub-diagonal of the attention
    matrix. Read the attention mass on that diagonal for the valid queries (i >= period) and
    average. attn: (..., T, T) -> (...); a perfect induction head scores 1.0. (Anthropic,
    In-context Learning and Induction Heads. See docs/hand-write/phase5-interp.md.)
    """
    raise NotImplementedError(
        "average the attention mass on the induction sub-diagonal over the valid queries"
    )
