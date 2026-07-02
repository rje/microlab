"""Hand-write exercise (Phase 11): reward-model preference loss and accuracy.

Fill in the ``NotImplementedError`` bodies so ``tests/exercises/test_phase11_reward.py`` passes.
Graded against ``microlab.model.reference.reward``. See docs/hand-write/phase11-reward.md.
"""

from __future__ import annotations

import torch


def preference_loss(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry preference loss: the negative log-likelihood, under the Bradley-Terry
    model, that the chosen sequence beats the rejected one, averaged over the batch. Drives
    the chosen reward above the rejected reward. (Christiano et al., RLHF from human
    preferences. See docs/hand-write/phase11-reward.md.)"""
    raise NotImplementedError(
        "turn the reward gap (chosen minus rejected) into a per-pair log-likelihood via the "
        "logistic function, then average — see the docstring"
    )


def reward_accuracy(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> float:
    """Fraction of pairs where the chosen sequence scores strictly higher than the rejected
    one (a tie is not a win). See docs/hand-write/phase11-reward.md."""
    raise NotImplementedError(
        "fraction of pairs whose chosen reward strictly exceeds the rejected reward"
    )
