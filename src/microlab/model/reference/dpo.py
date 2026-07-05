"""Reference DPO tools (Phase 12). `sequence_logprob` sums the response-token log-probs
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


def ipo_loss(
    policy_chosen_logp: torch.Tensor, policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor, ref_rejected_logp: torch.Tensor, beta: float = 0.5,
) -> tuple[torch.Tensor, float]:
    """IPO loss (Azar et al. 2024) + implicit-reward accuracy. Uses the same margin as DPO,
    h = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected), but where DPO's
    -logsigmoid(beta*h) keeps falling as h -> +inf (the over-optimization trap that lets the
    policy drift arbitrarily far from the reference), IPO regresses h to a FINITE target
    1/(2*beta): loss = (h - 1/(2*beta))^2. The target is a ceiling as well as a floor, so the
    policy is penalized for pushing the margin past it and cannot run away from the reference.
    Raising beta lowers the target margin, keeping the policy closer to the reference."""
    pi_logratios = policy_chosen_logp - policy_rejected_logp
    ref_logratios = ref_chosen_logp - ref_rejected_logp
    h = pi_logratios - ref_logratios
    target = 1.0 / (2.0 * beta)
    loss = ((h - target) ** 2).mean()
    # acc = fraction with chosen implicit reward > rejected; sign(reward gap) == sign(h).
    acc = (h > 0).float().mean().item()
    return loss, acc
