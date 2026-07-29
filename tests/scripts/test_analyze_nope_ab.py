"""scripts/analyze_nope_ab.py pure logic: TB val-loss extraction, the matched-step
val-loss table, the length-generalization comparison table, and the side-by-side passkey
grid. All CPU; the TB test writes a real tfevents file with SummaryWriter."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "analyze_nope_ab", Path(__file__).resolve().parents[2] / "scripts" / "analyze_nope_ab.py")
an = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(an)


def test_read_val_losses_from_tfevents(tmp_path):
    from torch.utils.tensorboard import SummaryWriter

    w = SummaryWriter(log_dir=str(tmp_path))
    for step, loss in [(250, 5.0), (500, 4.5), (750, 4.2)]:
        w.add_scalar("val/loss", loss, step)
        w.add_scalar("train/loss", loss - 0.1, step)  # decoy tag, must be ignored
    w.close()
    out = an.read_val_losses(tmp_path)
    assert set(out) == {250, 500, 750}
    assert out[500] == pytest.approx(4.5)


def test_read_val_losses_missing_events_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        an.read_val_losses(tmp_path)


def test_matched_step_table_shows_both_arms_and_delta():
    rope = {250: 5.00, 500: 4.50, 750: 4.20}
    nope = {250: 5.10, 500: 4.55, 1000: 4.00}  # 750/1000 don't match
    table = an.matched_step_table(rope, nope)
    steps = [line.split()[0] for line in table.splitlines()[1:]]
    assert steps == ["250", "500"]  # only steps BOTH arms evaluated, in order
    assert "5.0000" in table and "4.5500" in table
    assert "+0.1000" in table and "+0.0500" in table  # nope - rope


def test_matched_step_table_no_overlap_raises():
    with pytest.raises(ValueError, match="no matched steps"):
        an.matched_step_table({250: 5.0}, {500: 4.5})


def test_length_gen_table_pairs_lengths():
    def rep(losses):
        return {"loss": {"results": [
            {"length": length, "mean_loss": ml, "ppl": 2.0 ** ml}
            for length, ml in losses.items()]}}

    table = an.length_gen_table(rep({512: 3.0, 1024: 3.1, 2048: 5.0}),
                                rep({512: 3.05, 1024: 3.1, 2048: 3.3}))
    assert "512" in table and "2048" in table
    assert "3.0000" in table and "3.3000" in table
    assert "+0.0500" in table and "-1.7000" in table


def test_length_gen_table_mismatched_lengths_raise():
    a = {"loss": {"results": [{"length": 512, "mean_loss": 3.0, "ppl": 20.0}]}}
    b = {"loss": {"results": [{"length": 1024, "mean_loss": 3.0, "ppl": 20.0}]}}
    with pytest.raises(ValueError, match="lengths"):
        an.length_gen_table(a, b)


def test_passkey_pair_table_interleaves_arms():
    def cells(acc):
        return [{"length": length, "depth": d, "acc": acc, "n": 10,
                 "correct": int(acc * 10), "samples": []}
                for length in (512, 2048) for d in (0.1, 0.9)]

    table = an.passkey_pair_table(cells(1.0), cells(0.5))
    assert "512" in table and "2048" in table
    assert "1.00/0.50" in table  # rope/nope side by side in one cell
    assert "rope/nope" in table
