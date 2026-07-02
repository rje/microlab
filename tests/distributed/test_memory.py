"""Memory-budget oracle: the closed-form bookkeeping behind 'will it fit'."""

import pytest

from microlab.distributed.reference.memory import memory_budget

GB = 1e9
ONE_B = dict(n_params=1_000_000_000, n_layer=24, n_embd=1792, block_size=1024,
             micro_batch=16)


def test_single_gpu_baseline():
    b = memory_budget(**ONE_B)
    assert b["params"] == pytest.approx(2 * ONE_B["n_params"])
    assert b["optimizer"] == pytest.approx(12 * ONE_B["n_params"])
    assert b["total"] == sum(b[k] for k in ("params", "grads", "optimizer", "activations"))


def test_zero_stages_shed_state_monotonically():
    budgets = [memory_budget(**ONE_B, dp=8, zero_stage=z)["total"] for z in (0, 1, 2, 3)]
    assert budgets[0] > budgets[1] > budgets[2] > budgets[3]
    z3 = memory_budget(**ONE_B, dp=8, zero_stage=3)
    assert z3["params"] == pytest.approx(2 * ONE_B["n_params"] / 8)


def test_tp_divides_everything():
    a = memory_budget(**ONE_B)
    b = memory_budget(**ONE_B, tp=2)
    for key in ("params", "grads", "optimizer", "activations"):
        assert b[key] == pytest.approx(a[key] / 2)


def test_grad_checkpoint_slashes_activations():
    a = memory_budget(**ONE_B)["activations"]
    b = memory_budget(**ONE_B, grad_checkpoint=True)["activations"]
    assert b == pytest.approx(a / 34)


def test_dp_alone_shrinks_nothing_at_zero0():
    assert memory_budget(**ONE_B)["total"] == pytest.approx(
        memory_budget(**ONE_B, dp=8)["total"])
