"""Peri-LN-vs-Pre-LN A/B analysis: matched-step val-loss table + final val loss/ppl for
N run dirs (mirrors scripts/analyze_nope_ab.py, minus the length-gen legs — block
layout has no positional story to test).

    python scripts/analyze_periln_ab.py                       # the two default arms
    python scripts/analyze_periln_ab.py runs/periln-ab-pre runs/periln-ab-peri \\
        runs/periln-ab-pre-s1338 runs/periln-ab-peri-s1338    # multi-seed variance mode

Accepts ANY number of run dirs so the variance measurement (2-3 seeds per arm, run dirs
like runs/periln-ab-{pre,peri}-s<seed>) reuses this script unchanged: one column per
run. With EXACTLY two runs a delta column (second - first) is added — pass the pre arm
first, so delta = peri - pre and negative = Peri-LN better (lower loss).

Column labels are run-dir basenames; a run that hasn't produced val/loss points yet
RAISES (start the arm, or drop it from the list) rather than printing a hole.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def read_val_losses(run_dir: Path) -> dict[int, float]:
    """step -> val/loss from the run dir's tfevents file(s)."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    run_dir = Path(run_dir)
    if not list(run_dir.glob("events.out.tfevents.*")):
        raise FileNotFoundError(f"no tfevents in {run_dir} — has the arm started?")
    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    acc.Reload()
    if "val/loss" not in acc.Tags()["scalars"]:
        raise ValueError(f"no val/loss scalars in {run_dir} — first eval not reached yet")
    return {ev.step: ev.value for ev in acc.Scalars("val/loss")}


def matched_step_table(runs: dict[str, dict[int, float]]) -> str:
    """Val loss at every step ALL runs have evaluated, one column per run. With exactly
    two runs, a delta column (second - first) is appended."""
    losses = list(runs.values())
    steps = sorted(set.intersection(*(set(d) for d in losses)))
    if not steps:
        raise ValueError("no matched steps across the given runs")
    widths = [max(len(name) + 2, 10) for name in runs]
    header = f"{'step':<8}" + "".join(f"{name:<{w}}" for name, w in zip(runs, widths, strict=True))
    delta = len(runs) == 2
    if delta:
        header += "delta"
    lines = [header.rstrip()]
    for s in steps:
        row = f"{s:<8d}" + "".join(
            f"{d[s]:<{w}.4f}" for d, w in zip(losses, widths, strict=True))
        if delta:
            row += f"{losses[1][s] - losses[0][s]:<+10.4f}".rstrip()
        lines.append(row.rstrip())
    return "\n".join(lines)


def final_summary_table(runs: dict[str, dict[int, float]]) -> str:
    """Each run's LAST evaluated step with its val loss and ppl (= exp(loss))."""
    w = max(max(len(name) for name in runs) + 2, 10)
    lines = [f"{'run':<{w}}{'step':<8}{'val_loss':<10}ppl"]
    for name, d in runs.items():
        last = max(d)
        lines.append(f"{name:<{w}}{last:<8d}{d[last]:<10.4f}{math.exp(d[last]):<10.2f}"
                     .rstrip())
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*",
                    default=["runs/periln-ab-pre", "runs/periln-ab-peri"],
                    help="run dirs to compare (default: the two A/B arms; pass extra "
                         "seed dirs for the variance measurement)")
    args = ap.parse_args()

    names = [Path(r).name for r in args.runs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate run-dir basenames in {names}: pass distinct runs")
    runs = {name: read_val_losses(Path(r)) for name, r in zip(names, args.runs, strict=True)}

    print("== matched-step val loss (train length 1024, same seeded batches) ==")
    print(matched_step_table(runs))
    print("\n== final val loss / perplexity (each run at its own last eval step) ==")
    print(final_summary_table(runs))


if __name__ == "__main__":
    main()
