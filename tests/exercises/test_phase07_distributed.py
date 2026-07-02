"""Spec + validation for the hand-written Phase-7 memory budget."""

import itertools

import pytest

from microlab.distributed.reference.memory import memory_budget as ref_budget
from microlab.exercises.phase07_distributed import memory_budget


def test_memory_budget_matches_reference_across_matrix():
    base = dict(n_params=7_000_000_000, n_layer=32, n_embd=4096, block_size=2048,
                micro_batch=4)
    for dp, tp, pp, z, ck in itertools.product((1, 8), (1, 2), (1, 4), (0, 1, 2, 3),
                                               (False, True)):
        got = memory_budget(**base, dp=dp, tp=tp, pp=pp, zero_stage=z, grad_checkpoint=ck)
        want = ref_budget(**base, dp=dp, tp=tp, pp=pp, zero_stage=z, grad_checkpoint=ck)
        assert got == pytest.approx(want), (dp, tp, pp, z, ck)

pytestmark = pytest.mark.exercise
