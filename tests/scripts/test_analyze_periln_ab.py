"""scripts/analyze_periln_ab.py pure logic: TB val-loss extraction, the N-run
matched-step val-loss table (delta column only for exactly two runs — the plain A/B),
and the final loss/ppl summary. All CPU; the TB test writes a real tfevents file with
SummaryWriter."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "analyze_periln_ab",
    Path(__file__).resolve().parents[2] / "scripts" / "analyze_periln_ab.py")
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


def test_matched_step_table_two_runs_has_delta():
    runs = {"periln-ab-pre": {250: 5.00, 500: 4.50, 750: 4.20},
            "periln-ab-peri": {250: 5.10, 500: 4.45, 1000: 4.00}}  # 750/1000 unmatched
    table = an.matched_step_table(runs)
    steps = [line.split()[0] for line in table.splitlines()[1:]]
    assert steps == ["250", "500"]  # only steps ALL runs evaluated, in order
    assert "periln-ab-pre" in table and "periln-ab-peri" in table
    assert "5.0000" in table and "4.4500" in table
    assert "+0.1000" in table and "-0.0500" in table  # delta = second - first (peri-pre)


def test_matched_step_table_many_runs_no_delta():
    # Multi-seed variance mode: one column per run dir, no pairwise delta.
    runs = {f"pre-s{s}": {250: 5.0 + s / 100, 500: 4.5} for s in range(3)}
    table = an.matched_step_table(runs)
    assert "delta" not in table
    assert "5.0100" in table and "5.0200" in table


def test_matched_step_table_no_overlap_raises():
    with pytest.raises(ValueError, match="no matched steps"):
        an.matched_step_table({"a": {250: 5.0}, "b": {500: 4.5}})


def test_final_summary_table_reports_last_step_loss_and_ppl():
    runs = {"periln-ab-pre": {250: 5.0, 500: 3.0},
            "periln-ab-peri": {250: 4.9, 750: 2.5}}
    table = an.final_summary_table(runs)
    assert "500" in table and "750" in table  # each run's own last evaluated step
    assert f"{math.exp(3.0):.2f}" in table
    assert f"{math.exp(2.5):.2f}" in table
