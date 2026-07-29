"""NoPE-vs-RoPE A/B analysis: matched-step val-loss table, length-generalization
comparison, and side-by-side passkey grids.

    python scripts/analyze_nope_ab.py

Reads (defaults, all overridable):
  - runs/nope-ab-rope, runs/nope-ab-nope        TB events -> matched-step val loss
  - evals/length_gen/nope-ab-{rope,nope}.json   scripts/eval_length_gen.py output

If a length-gen JSON is missing this RAISES with the exact command to produce it (run
each arm's eval after training finishes; ~5 min each on the RTX 6000 Ada):

    python scripts/eval_length_gen.py --run runs/nope-ab-rope \\
        --out evals/length_gen/nope-ab-rope.json
    python scripts/eval_length_gen.py --run runs/nope-ab-nope \\
        --out evals/length_gen/nope-ab-nope.json

Deltas are ALWAYS nope - rope: negative = NoPE better (lower loss), and in the passkey
grid each cell is "rope_acc/nope_acc".
"""

from __future__ import annotations

import argparse
import json
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


def matched_step_table(rope: dict[int, float], nope: dict[int, float]) -> str:
    """Val loss at every step both arms have evaluated; delta = nope - rope."""
    steps = sorted(set(rope) & set(nope))
    if not steps:
        raise ValueError("no matched steps between the two arms")
    lines = [f"{'step':<8}{'rope':<10}{'nope':<10}{'nope-rope':<10}"]
    for s in steps:
        lines.append(f"{s:<8d}{rope[s]:<10.4f}{nope[s]:<10.4f}{nope[s] - rope[s]:<+10.4f}")
    return "\n".join(lines)


def length_gen_table(rope_report: dict, nope_report: dict) -> str:
    """Val loss/ppl per eval length from two eval_length_gen JSON reports."""
    a = {r["length"]: r for r in rope_report["loss"]["results"]}
    b = {r["length"]: r for r in nope_report["loss"]["results"]}
    if set(a) != set(b):
        raise ValueError(f"arms evaluated different lengths: {sorted(a)} vs {sorted(b)}")
    lines = [f"{'length':<8}{'rope_loss':<11}{'nope_loss':<11}{'nope-rope':<11}"
             f"{'rope_ppl':<10}{'nope_ppl':<10}"]
    for length in sorted(a):
        ra, rb = a[length], b[length]
        lines.append(
            f"{length:<8d}{ra['mean_loss']:<11.4f}{rb['mean_loss']:<11.4f}"
            f"{rb['mean_loss'] - ra['mean_loss']:<+11.4f}"
            f"{ra['ppl']:<10.2f}{rb['ppl']:<10.2f}")
    return "\n".join(lines)


def passkey_pair_table(rope_cells: list[dict], nope_cells: list[dict]) -> str:
    """One grid, each cell 'rope_acc/nope_acc' (lengths down, depths across)."""
    ra = {(c["length"], c["depth"]): c["acc"] for c in rope_cells}
    na = {(c["length"], c["depth"]): c["acc"] for c in nope_cells}
    lengths = sorted({k[0] for k in ra} | {k[0] for k in na})
    depths = sorted({k[1] for k in ra} | {k[1] for k in na})
    lines = ["passkey acc rope/nope",
             "length  " + "".join(f"depth={d:<10g}" for d in depths)]
    for length in lengths:
        row = f"{length:<8d}"
        for d in depths:
            r, n = ra.get((length, d)), na.get((length, d))
            cell = ("-" if r is None else f"{r:.2f}") + "/" + \
                   ("-" if n is None else f"{n:.2f}")
            row += f"{cell:<16}"
        lines.append(row.rstrip())
    return "\n".join(lines)


def _load_eval_json(path: Path, run: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — produce it with:\n"
            f"  python scripts/eval_length_gen.py --run {run} --out {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rope-run", default="runs/nope-ab-rope")
    ap.add_argument("--nope-run", default="runs/nope-ab-nope")
    ap.add_argument("--rope-eval", type=Path,
                    default=Path("evals/length_gen/nope-ab-rope.json"))
    ap.add_argument("--nope-eval", type=Path,
                    default=Path("evals/length_gen/nope-ab-nope.json"))
    args = ap.parse_args()

    print("== matched-step val loss (train length 1024, same seeded batches) ==")
    print(matched_step_table(read_val_losses(Path(args.rope_run)),
                             read_val_losses(Path(args.nope_run))))

    rope_report = _load_eval_json(args.rope_eval, args.rope_run)
    nope_report = _load_eval_json(args.nope_eval, args.nope_run)
    print(f"\n== length generalization (val loss vs eval length; steps "
          f"{rope_report['step']}/{nope_report['step']}) ==")
    print(length_gen_table(rope_report, nope_report))
    print("\n== passkey retrieval ==")
    print(passkey_pair_table(rope_report["passkey"]["cells"],
                             nope_report["passkey"]["cells"]))


if __name__ == "__main__":
    main()
