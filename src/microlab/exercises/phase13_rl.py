"""Hand-write exercise (Phase 13): verifiable reward, GRPO advantages, and the PPO clip loss.

Fill in the ``NotImplementedError`` bodies so ``tests/exercises/test_phase13_rl.py`` passes.
Graded against ``microlab.model.reference.rl``. See docs/hand-write/phase13-rl.md.

``extract_answer`` (parsing "#### N" or the last number in a string) is already implemented
in ``microlab.model.reference.rl`` -- reuse it via import, don't reimplement it.
"""

from __future__ import annotations

import torch


def verifiable_reward(generated: str, gold: str) -> float:
    """1.0 if the answer extracted from `generated` equals the gold answer, else 0.0. Reuse
    ``extract_answer`` (per the module docstring) on both sides; a failed extraction (None)
    never matches. See docs/hand-write/phase13-rl.md."""
    raise NotImplementedError(
        "extract the answer from each string and compare — a None extraction never counts"
    )


def group_normalized_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """GRPO advantage: within a group of completions for the same prompt, standardize the
    rewards to zero mean and unit scale (eps guards the std) — the group is its own baseline,
    so GRPO needs no learned critic. See docs/hand-write/phase13-rl.md."""
    raise NotImplementedError(
        "standardize the group's rewards to zero mean / unit scale (eps guards the std)"
    )


def ppo_clip_loss(
    logprobs: torch.Tensor, old_logprobs: torch.Tensor, advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """PPO clipped surrogate loss (a scalar to MINIMIZE). Form the likelihood ratio between
    the new policy and the sampling policy on the sampled tokens, weight it by the advantages,
    and clip the ratio into a trust region so an off-policy step can't chase the advantage
    arbitrarily far. The clip is the whole point. (Schulman et al., PPO. See
    docs/hand-write/phase13-rl.md.)"""
    raise NotImplementedError(
        "build the probability ratio from the log-prob difference, form the clipped and "
        "unclipped advantage-weighted terms, and combine them with PPO's pessimistic "
        "surrogate (then negate and average) — the clip bounds and sign are yours to derive"
    )
