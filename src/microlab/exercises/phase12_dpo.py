"""Hand-write exercise (Phase 12): DPO sequence log-probs and the DPO loss.

Fill in the ``NotImplementedError`` bodies so ``tests/exercises/test_phase12_dpo.py`` passes.
Graded against ``microlab.model.reference.dpo``. See docs/hand-write/phase12-dpo.md.
"""

from __future__ import annotations

import torch

IGNORE_INDEX = -100


def sequence_logprob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum of per-token log-probs of `labels` under `logits`, over the supervised
    (non-IGNORE_INDEX) response tokens. Causal shift: logits[:, :-1] predict labels[:, 1:].
    Returns (B,). The subtlety is picking out each label's log-prob while the IGNORE_INDEX
    positions (a negative sentinel) are still in the tensor — find an indexing + masking
    scheme that doesn't crash on them. See docs/hand-write/phase12-dpo.md."""
    raise NotImplementedError(
        "log-softmax the shifted logits, select each label's log-prob, zero out the "
        "IGNORE_INDEX positions, and sum over the sequence — the indexing/masking detail is "
        "yours to work out"
    )


def dpo_loss(
    policy_chosen_logp: torch.Tensor, policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor, ref_rejected_logp: torch.Tensor, beta: float = 0.1,
) -> tuple[torch.Tensor, float]:
    """DPO loss + implicit-reward accuracy. The policy is pushed to increase
    (logp_chosen - logp_rejected) beyond the reference's, scaled by beta. Returns
    (loss, accuracy). (Rafailov et al., Direct Preference Optimization. See
    docs/hand-write/phase12-dpo.md.)"""
    raise NotImplementedError(
        "form the policy and reference log-ratios (chosen minus rejected), scale beta times "
        "their difference, and apply the same Bradley-Terry logistic loss as Phase 11; "
        "accuracy compares the per-response implicit rewards (beta * policy-vs-reference "
        "log-prob gap) of chosen vs rejected"
    )
