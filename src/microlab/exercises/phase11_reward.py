"""Hand-write exercise (Phase 11): reward-model preference loss and accuracy.

Fill in the ``NotImplementedError`` bodies so ``tests/exercises/test_phase11_reward.py`` passes.
Graded against ``microlab.model.reference.reward``. See docs/hand-write/phase11-reward.md.
"""

from __future__ import annotations

import torch


def preference_loss(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry loss: -log sigmoid(r_chosen - r_rejected), averaged. Drives the
    chosen reward above the rejected reward."""
    raise NotImplementedError("-F.logsigmoid(chosen_rewards - rejected_rewards).mean()")


def reward_accuracy(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> float:
    """Fraction of pairs where the chosen sequence scores higher than the rejected one."""
    raise NotImplementedError("(chosen_rewards > rejected_rewards).float().mean().item()")
