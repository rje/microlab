"""Reference RL-on-verifiable-tasks tools (Phase 10). Verifiable reward = did the model's
answer match the gold answer (checkable, no learned reward model). GRPO normalizes
rewards within a sampled group into advantages; the PPO clipped objective turns advantages
into a policy-gradient loss. The oracle the owner diffs their hand-written versions against."""

from __future__ import annotations

import re

import torch


def extract_answer(text: str) -> str | None:
    """Pull the final answer from a GSM8K-style solution: the number after '####' if
    present, else the last number in the text. Returns a normalized string or None."""
    m = re.search(r"####\s*(-?[0-9][0-9,]*\.?[0-9]*)", text)
    if not m:
        nums = re.findall(r"-?[0-9][0-9,]*\.?[0-9]*", text)
        if not nums:
            return None
        m_str = nums[-1]
    else:
        m_str = m.group(1)
    return m_str.replace(",", "").rstrip(".")


def verifiable_reward(generated: str, gold: str) -> float:
    """1.0 if the extracted answer equals the gold answer, else 0.0."""
    pred = extract_answer(generated)
    return 1.0 if pred is not None and pred == extract_answer(gold) else 0.0


def group_normalized_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """GRPO advantage: within a group of completions for the same prompt, A_i =
    (r_i - mean) / (std + eps). Zero-mean, unit-scale — the group is its own baseline."""
    return (rewards - rewards.mean()) / (rewards.std() + eps)


def ppo_clip_loss(
    logprobs: torch.Tensor, old_logprobs: torch.Tensor, advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """PPO clipped surrogate loss (to MINIMIZE): -mean(min(ratio*A, clip(ratio,1±eps)*A)),
    where ratio = exp(logprobs - old_logprobs). Clipping removes the incentive to move the
    policy too far from the sampling policy."""
    ratio = torch.exp(logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -torch.min(unclipped, clipped).mean()


def kl_to_reference(logprobs: torch.Tensor, ref_logprobs: torch.Tensor) -> torch.Tensor:
    """Mean KL(policy || ref) estimator (the penalty that keeps RL near the SFT model)."""
    return (logprobs - ref_logprobs).mean()
