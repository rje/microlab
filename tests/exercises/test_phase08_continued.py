"""Spec + validation for the hand-written Phase-8 continued-pretraining primitives.

Implement ``microlab.exercises.phase08_continued`` until these pass. The last test diffs your
``build_replay_mix`` against the reference oracle over randomized fractions.
"""

import pytest
import torch

from microlab.exercises.phase08_continued import (
    build_replay_mix,
    forgetting_score,
    interpolated_rope_cache,
)
from microlab.model.reference.continued import build_replay_mix as ref_mix
from microlab.model.reference.continued import interpolated_rope_cache as ref_cache


def test_forgetting_score_sign():
    assert forgetting_score(3.0, 3.5) == pytest.approx(0.5)   # forgot
    assert forgetting_score(3.0, 3.0) == pytest.approx(0.0)   # retained
    assert forgetting_score(3.0, 2.7) == pytest.approx(-0.3)  # improved too


def test_replay_mix_zero_is_identity():
    new = torch.arange(100)
    assert torch.equal(build_replay_mix(new, torch.arange(50), 0.0), new)


def test_replay_mix_fraction_is_correct():
    new = torch.arange(900)
    old = torch.arange(10000, 20000)
    mixed = build_replay_mix(new, old, 0.1)
    n_old = (mixed >= 10000).sum().item()
    assert n_old / len(mixed) == pytest.approx(0.1, abs=0.01)


def test_replay_mix_caps_at_available_old():
    mixed = build_replay_mix(torch.arange(1000), torch.arange(10000, 10005), 0.9)
    assert len(mixed) == 1005


def test_replay_mix_matches_reference_oracle():
    new = torch.arange(500)
    old = torch.arange(10000, 13000)
    for f in [0.0, 0.05, 0.1, 0.25, 0.5, 0.75]:
        assert torch.equal(build_replay_mix(new, old, f), ref_mix(new, old, f)), f


def test_interpolated_rope_cache_matches_reference():
    for scale in (1.0, 2.0, 4.0):
        a_cos, a_sin = interpolated_rope_cache(64, 8, scale)
        b_cos, b_sin = ref_cache(64, 8, scale)
        assert torch.allclose(a_cos, b_cos, atol=1e-6)
        assert torch.allclose(a_sin, b_sin, atol=1e-6)

pytestmark = pytest.mark.exercise
