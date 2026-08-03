"""Batch geometry must be invariant to world size, and must refuse layouts that are not.

This is the arithmetic behind a migratable run. If `grad_accum` were a plain config knob,
moving a run from 1 GPU to 4 would multiply the effective batch by 4 — the LR schedule and
the token accounting would both be wrong, and the loss curve would still look plausible.
"""

from __future__ import annotations

import pytest

from microlab.train.distributed import batch_geometry

TPS, BLOCK = 524_288, 32_768          # the 1B capstone's global batch


def test_the_capstone_geometry_across_world_sizes():
    """16 sequences per step, however they are divided."""
    for ws, expect_accum in ((1, 16), (2, 8), (4, 4), (8, 2), (16, 1)):
        seqs, accum = batch_geometry(TPS, BLOCK, batch_size=1, ws=ws)
        assert seqs == 16, ws
        assert accum == expect_accum, ws
        # the invariant that matters: total sequences consumed per step is constant
        assert ws * 1 * accum == seqs


@pytest.mark.parametrize("bs", [1, 2, 4])
def test_batch_size_participates_in_the_division(bs):
    seqs, accum = batch_geometry(TPS, BLOCK, batch_size=bs, ws=4)
    assert seqs == 16
    assert 4 * bs * accum == seqs


def test_indivisible_world_size_raises_rather_than_rounding():
    """16 sequences cannot be split across 5 ranks. Rounding would silently resize the
    batch on exactly the migration where it matters most."""
    with pytest.raises(ValueError, match="does not divide"):
        batch_geometry(TPS, BLOCK, batch_size=1, ws=5)


def test_indivisible_batch_size_raises():
    with pytest.raises(ValueError, match="does not divide"):
        batch_geometry(TPS, BLOCK, batch_size=3, ws=4)


def test_tokens_not_divisible_by_block_raises():
    with pytest.raises(ValueError, match="not divisible by block_size"):
        batch_geometry(500_000, BLOCK, batch_size=1, ws=1)


def test_error_names_the_fix():
    """A geometry error should say what to change, not just that it failed."""
    with pytest.raises(ValueError) as e:
        batch_geometry(TPS, BLOCK, batch_size=1, ws=5)
    assert "tokens_per_step" in str(e.value)


def test_world_size_8_matches_the_cloud_plan():
    """The 4x and 8x configurations the cloud plan prices must actually be expressible."""
    for ws in (4, 8):
        seqs, accum = batch_geometry(TPS, BLOCK, batch_size=1, ws=ws)
        assert seqs == 16 and accum * ws == 16
