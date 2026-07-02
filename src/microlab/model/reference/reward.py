"""Reference reward-model tools (Phase 11): a scalar reward head over the GPT, and the
Bradley-Terry pairwise preference loss used to train reward models in RLHF. The oracle
the owner diffs their hand-written preference loss against."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F


class RewardModel(nn.Module):
    """Wraps a GPT with a scalar value head. The reward for a sequence is the value at
    its last token (reuses the GPT's transformer trunk; the LM head is unused)."""

    def __init__(self, gpt: nn.Module) -> None:
        super().__init__()
        self.gpt = gpt
        self.value_head = nn.Linear(gpt.config.n_embd, 1, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        t = self.gpt.transformer
        pos = torch.arange(idx.size(1), device=idx.device)
        x = t.drop(t.wte(idx) + t.wpe(pos))
        for block in t.h:
            x = block(x)
        x = t.ln_f(x)
        return self.value_head(x).squeeze(-1)  # (B, T) scalar per position

    def sequence_reward(self, idx: torch.Tensor) -> torch.Tensor:
        """Scalar reward per sequence = value at the final token. (B,)"""
        return self.forward(idx)[:, -1]


def preference_loss(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry loss: -log sigmoid(r_chosen - r_rejected), averaged. Drives the
    chosen reward above the rejected reward."""
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()


def reward_accuracy(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> float:
    """Fraction of pairs where the chosen sequence scores higher than the rejected one."""
    return (chosen_rewards > rejected_rewards).float().mean().item()
