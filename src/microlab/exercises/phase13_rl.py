"""Hand-write exercise (Phase 13): verifiable reward, GRPO advantages, and the PPO clip loss.

Fill in the ``NotImplementedError`` bodies so ``tests/exercises/test_phase13_rl.py`` passes.
Graded against ``microlab.model.reference.rl``. See docs/hand-write/phase13-rl.md.

``extract_answer`` (parsing "#### N" or the last number in a string) is already implemented
in ``microlab.model.reference.rl`` -- reuse it via import, don't reimplement it.
"""

from __future__ import annotations

import torch


def verifiable_reward(generated: str, gold: str) -> float:
    """1.0 if the extracted answer equals the gold answer, else 0.0."""
    raise NotImplementedError(
        "from microlab.model.reference.rl import extract_answer; "
        "pred = extract_answer(generated); "
        "return 1.0 if pred is not None and pred == extract_answer(gold) else 0.0"
    )


def group_normalized_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """GRPO advantage: within a group of completions for the same prompt, A_i =
    (r_i - mean) / (std + eps). Zero-mean, unit-scale — the group is its own baseline."""
    raise NotImplementedError("(rewards - rewards.mean()) / (rewards.std() + eps)")


def ppo_clip_loss(
    logprobs: torch.Tensor, old_logprobs: torch.Tensor, advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """PPO clipped surrogate loss (to MINIMIZE): -mean(min(ratio*A, clip(ratio,1±eps)*A)),
    where ratio = exp(logprobs - old_logprobs). Clipping removes the incentive to move the
    policy too far from the sampling policy."""
    raise NotImplementedError(
        "ratio = torch.exp(logprobs - old_logprobs); "
        "unclipped = ratio * advantages; "
        "clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages; "
        "return -torch.min(unclipped, clipped).mean()"
    )
