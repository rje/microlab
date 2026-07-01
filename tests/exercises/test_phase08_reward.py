"""Spec + validation for the hand-written Phase-8 reward-model preference loss.

Implement ``microlab.exercises.phase08_reward`` until these pass. Differential tests grade you
against ``microlab.model.reference.reward``.
"""

import math

import pytest
import torch

from microlab.exercises.phase08_reward import preference_loss, reward_accuracy
from microlab.model.reference.reward import preference_loss as ref_loss
from microlab.model.reference.reward import reward_accuracy as ref_acc


def test_preference_loss_known_value_equal_rewards():
    # equal rewards -> -log sigmoid(0) = log 2
    loss = preference_loss(torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]))
    assert loss.item() == pytest.approx(math.log(2), abs=1e-5)


def test_preference_loss_matches_reference():
    torch.manual_seed(0)
    chosen = torch.randn(16)
    rejected = torch.randn(16)
    assert preference_loss(chosen, rejected).item() == pytest.approx(
        ref_loss(chosen, rejected).item(), abs=1e-6
    )


def test_reward_accuracy_matches_reference():
    torch.manual_seed(1)
    chosen = torch.randn(32)
    rejected = torch.randn(32)
    assert reward_accuracy(chosen, rejected) == pytest.approx(ref_acc(chosen, rejected))


def test_reward_accuracy_known_value():
    c = torch.tensor([1.0, 2.0, -1.0])
    r = torch.tensor([0.0, 3.0, -2.0])
    assert reward_accuracy(c, r) == pytest.approx(2 / 3)

pytestmark = pytest.mark.exercise
