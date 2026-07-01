"""Spec + validation for the hand-written Phase-10 RL-on-verifiable-tasks primitives.

Implement ``microlab.exercises.phase10_rl`` until these pass. Differential tests grade you
against ``microlab.model.reference.rl``.
"""

import pytest
import torch

from microlab.exercises.phase10_rl import (
    group_normalized_advantages,
    ppo_clip_loss,
    verifiable_reward,
)
from microlab.model.reference.rl import extract_answer
from microlab.model.reference.rl import group_normalized_advantages as ref_advantages
from microlab.model.reference.rl import ppo_clip_loss as ref_ppo_loss
from microlab.model.reference.rl import verifiable_reward as ref_reward


def test_verifiable_reward_matches_reference():
    pairs = [
        ("the answer is #### 18", "#### 18"),
        ("so it must be 18", "#### 18"),
        ("#### 17", "#### 18"),
        ("nope", "#### 18"),
    ]
    for gen, gold in pairs:
        assert verifiable_reward(gen, gold) == ref_reward(gen, gold)


def test_verifiable_reward_known_values():
    assert verifiable_reward("#### 18", "#### 18") == 1.0
    assert verifiable_reward("#### 17", "#### 18") == 0.0
    assert extract_answer("#### 18") == "18"  # sanity: the given helper is unchanged


def test_group_advantages_matches_reference():
    torch.manual_seed(0)
    rewards = torch.randn(8)
    assert torch.allclose(group_normalized_advantages(rewards), ref_advantages(rewards), atol=1e-6)


def test_group_advantages_mean_zero():
    a = group_normalized_advantages(torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert a.mean().abs() < 1e-5


def test_ppo_clip_loss_matches_reference():
    torch.manual_seed(0)
    lp, old_lp = torch.randn(8), torch.randn(8)
    adv = torch.randn(8)
    assert ppo_clip_loss(lp, old_lp, adv).item() == pytest.approx(
        ref_ppo_loss(lp, old_lp, adv).item(), abs=1e-6
    )


def test_ppo_clip_known_value():
    # ratio = e, A = 1, eps = 0.2 -> clipped branch wins -> loss = -(1+eps)*A = -1.2
    old = torch.zeros(1)
    new = torch.tensor([1.0])
    adv = torch.tensor([1.0])
    assert ppo_clip_loss(new, old, adv, clip_eps=0.2).item() == pytest.approx(-1.2, abs=1e-5)

pytestmark = pytest.mark.exercise
