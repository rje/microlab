"""Spec + validation for the hand-written Phase-12 DPO primitives.

Implement ``microlab.exercises.phase12_dpo`` until these pass. Differential tests grade you
against ``microlab.model.reference.dpo``.
"""

import math

import pytest
import torch

from microlab.exercises.phase12_dpo import IGNORE_INDEX, dpo_loss, sequence_logprob
from microlab.model.reference.dpo import IGNORE_INDEX as REF_IGNORE_INDEX
from microlab.model.reference.dpo import dpo_loss as ref_dpo_loss
from microlab.model.reference.dpo import sequence_logprob as ref_sequence_logprob


def test_ignore_index_matches_reference():
    assert IGNORE_INDEX == REF_IGNORE_INDEX


def test_sequence_logprob_matches_reference():
    torch.manual_seed(0)
    logits = torch.randn(3, 6, 12)
    labels = torch.randint(0, 12, (3, 6))
    labels[:, 0] = IGNORE_INDEX  # first position is never a supervised prediction target
    labels[1, 2] = IGNORE_INDEX  # extra mask mid-sequence
    got = sequence_logprob(logits, labels)
    want = ref_sequence_logprob(logits, labels)
    assert torch.allclose(got, want, atol=1e-5)


def test_sequence_logprob_known_value():
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 8)
    labels = torch.tensor([[IGNORE_INDEX, 2, 5, 1]])
    lp = torch.log_softmax(logits[:, :-1], dim=-1)
    expected = lp[0, 0, 2] + lp[0, 1, 5] + lp[0, 2, 1]
    assert sequence_logprob(logits, labels).item() == pytest.approx(expected.item(), abs=1e-5)


def test_dpo_loss_zero_advantage_is_log2():
    z = torch.zeros(4)
    loss, _ = dpo_loss(z, z, z, z, beta=0.1)  # policy == ref -> logits 0 -> -log sigmoid 0
    assert loss.item() == pytest.approx(math.log(2), abs=1e-5)


def test_dpo_loss_matches_reference():
    torch.manual_seed(1)
    pc, pr = torch.randn(8), torch.randn(8)
    rc, rr = torch.randn(8), torch.randn(8)
    loss, acc = dpo_loss(pc, pr, rc, rr, beta=0.3)
    rloss, racc = ref_dpo_loss(pc, pr, rc, rr, beta=0.3)
    assert loss.item() == pytest.approx(rloss.item(), abs=1e-6)
    assert acc == pytest.approx(racc)

pytestmark = pytest.mark.exercise
