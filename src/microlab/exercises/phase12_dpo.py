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
    Returns (B,)."""
    raise NotImplementedError(
        "shift: logits[:, :-1], labels[:, 1:]; log_softmax(logits); gather at "
        "labels.clamp(min=0); mask out IGNORE_INDEX positions; sum over the sequence dim"
    )


def dpo_loss(
    policy_chosen_logp: torch.Tensor, policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor, ref_rejected_logp: torch.Tensor, beta: float = 0.1,
) -> tuple[torch.Tensor, float]:
    """DPO loss + implicit-reward accuracy. The policy is pushed to increase
    (logp_chosen - logp_rejected) beyond the reference's, scaled by beta."""
    raise NotImplementedError(
        "pi_logratios = policy_chosen_logp - policy_rejected_logp; "
        "ref_logratios = ref_chosen_logp - ref_rejected_logp; "
        "loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean(); "
        "acc = mean(beta*(policy_chosen_logp-ref_chosen_logp) > "
        "beta*(policy_rejected_logp-ref_rejected_logp))"
    )
