"""Reference DPO tools (Phase 9). `sequence_logprob` sums the response-token log-probs
under a model; `dpo_loss` is the Direct Preference Optimization objective, which raises
the policy's log-ratio on chosen vs rejected relative to a frozen reference model."""

from __future__ import annotations

import torch
from torch.nn import functional as F

IGNORE_INDEX = -100


def sequence_logprob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum of per-token log-probs of `labels` under `logits`, over the supervised
    (non-IGNORE_INDEX) response tokens. Causal shift: logits[:, :-1] predict labels[:, 1:].
    Returns (B,)."""
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    logp = F.log_softmax(logits, dim=-1)
    mask = labels != IGNORE_INDEX
    gathered = logp.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (gathered * mask).sum(dim=-1)


def dpo_loss(
    policy_chosen_logp: torch.Tensor, policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor, ref_rejected_logp: torch.Tensor, beta: float = 0.1,
) -> tuple[torch.Tensor, float]:
    """DPO loss + implicit-reward accuracy. The policy is pushed to increase
    (logp_chosen - logp_rejected) beyond the reference's, scaled by beta."""
    pi_logratios = policy_chosen_logp - policy_rejected_logp
    ref_logratios = ref_chosen_logp - ref_rejected_logp
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits).mean()
    chosen_reward = beta * (policy_chosen_logp - ref_chosen_logp)
    rejected_reward = beta * (policy_rejected_logp - ref_rejected_logp)
    acc = (chosen_reward > rejected_reward).float().mean().item()
    return loss, acc
