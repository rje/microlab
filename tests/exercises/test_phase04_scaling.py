"""Spec + validation for the hand-written Phase-4 scaling tools.

Implement ``microlab.exercises.phase04_scaling`` until these pass. ``count_params`` is graded
against the real model; ``training_flops_per_token`` against the reference; and
``fit_scaling_law`` by recovering a known power law.
"""

import pytest

from microlab.exercises.phase04_scaling import (
    count_params,
    fit_scaling_law,
    mup_attn_scale,
    mup_multipliers,
    training_flops_per_token,
)
from microlab.model.reference import scaling as ref
from microlab.model.reference.gpt import GPT, GPTConfig
from microlab.model.reference.scaling import (
    training_flops_per_token as ref_flops,
)

_CONFIGS = [
    GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32),
    GPTConfig(vocab_size=512, block_size=128, n_layer=4, n_head=4, n_embd=192),
    GPTConfig(vocab_size=100, block_size=64, n_layer=3, n_head=3, n_embd=96, bias=False),
]


@pytest.mark.parametrize("cfg", _CONFIGS)
def test_count_params_matches_real_model(cfg):
    # the real model is the oracle: your formula must reproduce it exactly
    assert count_params(cfg) == GPT(cfg).num_params()


@pytest.mark.parametrize("cfg", _CONFIGS)
def test_flops_matches_reference(cfg):
    assert training_flops_per_token(cfg) == ref_flops(cfg)


def test_fit_recovers_known_power_law():
    A_true, alpha_true = 3.0, 0.1
    params = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    losses = [A_true * n ** (-alpha_true) for n in params]
    A, alpha = fit_scaling_law(params, losses)
    assert A == pytest.approx(A_true, rel=1e-3)
    assert alpha == pytest.approx(alpha_true, rel=1e-3)


def test_mup_multipliers_matches_reference():
    for base, w in [(128, 128), (128, 512), (256, 1024)]:
        assert mup_multipliers(base, w) == pytest.approx(ref.mup_multipliers(base, w))
    assert mup_attn_scale(48) == pytest.approx(ref.mup_attn_scale(48))


pytestmark = pytest.mark.exercise
