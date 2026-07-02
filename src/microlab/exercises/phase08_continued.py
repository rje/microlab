"""Hand-write exercise (Phase 8): the three primitives of continued pretraining —
the forgetting metric, replay mixing, and RoPE position interpolation.

Fill in the ``NotImplementedError`` bodies (forgetting_score, build_replay_mix,
interpolated_rope_cache) so ``tests/exercises/test_phase08_continued.py`` passes. They're
graded against ``microlab.model.reference.continued``. See docs/hand-write/phase8-continued.md.
"""

from __future__ import annotations

import torch


def forgetting_score(loss_before: float, loss_after: float) -> float:
    """Increase in loss on the ORIGINAL domain after continued training on a new domain.
    Positive = forgot; ~0 = retained; negative = improved there too."""
    raise NotImplementedError("return how much the original-domain loss went UP")


def build_replay_mix(
    new_tokens: torch.Tensor, old_tokens: torch.Tensor, replay_fraction: float
) -> torch.Tensor:
    """Combine new-domain tokens with a `replay_fraction` slice of old-domain tokens so
    continued training rehearses the old domain. `replay_fraction` is the target SHARE of
    old tokens in the result (0.0 -> just the new tokens; must be < 1.0). If the old
    corpus is too small to hit the target, use all of it.

    Hint: to make old tokens `f` of the total, you need `n_old = f/(1-f) * n_new` of them.
    """
    raise NotImplementedError("build the replay mixture (see the fraction hint)")


def interpolated_rope_cache(seq_len: int, head_dim: int, scale: float, base: float = 10000.0):
    """Position interpolation: build_rope_cache but with positions t/scale. Returns
    (cos, sin) of shape (seq_len, head_dim//2). Graded vs the reference."""
    raise NotImplementedError("theta as in build_rope_cache; t = arange(seq_len)/scale")
