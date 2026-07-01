"""Spec + validation for the hand-written Phase-6 SFT primitives.

Implement ``microlab.exercises.phase06_sft`` until these pass. The differential tests grade
your masking + masked loss against ``microlab.model.reference.sft``.
"""

import pytest
import torch

from microlab.exercises.phase06_sft import IGNORE_INDEX, build_sft_example, masked_cross_entropy
from microlab.model.reference.sft import build_sft_example as ref_build
from microlab.model.reference.sft import masked_cross_entropy as ref_mce


class _FakeTok:
    def encode(self, s):
        return list(s.encode("utf-8"))


def test_build_sft_example_masks_prompt():
    tok = _FakeTok()
    input_ids, labels = build_sft_example(tok, "PROMPT>", "RESP")
    n = len(tok.encode("PROMPT>"))
    assert input_ids == tok.encode("PROMPT>") + tok.encode("RESP")
    assert labels[:n] == [IGNORE_INDEX] * n
    assert labels[n:] == tok.encode("RESP")


def test_build_sft_example_matches_reference():
    tok = _FakeTok()
    for p, r in [("Q:", "A"), ("### Response:\n", "hello world"), ("x", "y")]:
        assert build_sft_example(tok, p, r) == ref_build(tok, p, r)


def test_masked_cross_entropy_matches_reference():
    torch.manual_seed(0)
    logits = torch.randn(2, 6, 16)
    labels = torch.tensor(
        [[IGNORE_INDEX, IGNORE_INDEX, 3, 7, 1, 4], [IGNORE_INDEX, 2, 5, 8, 9, IGNORE_INDEX]]
    )
    assert torch.allclose(masked_cross_entropy(logits, labels), ref_mce(logits, labels))


def test_masked_cross_entropy_ignores_masked_positions():
    torch.manual_seed(1)
    logits = torch.randn(1, 5, 12)
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 3, 7, 1]])
    base = masked_cross_entropy(logits, labels)
    labels2 = labels.clone()
    labels2[0, 0] = 5  # a masked (dropped-by-shift) position -> no effect
    assert torch.allclose(base, masked_cross_entropy(logits, labels2))

pytestmark = pytest.mark.exercise
