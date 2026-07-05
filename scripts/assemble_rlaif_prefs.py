"""RLAIF stage 3 — combine candidate records (build_rlaif_candidates.py) with judge verdicts
into {prompt, chosen, rejected} pairs for dpo.py --loss ipo. Each verdict names, per record
index, the judged best and worst candidate; chosen = candidates[best], rejected = candidates
[worst]. Verdicts live in one or more JSON files (a list of {"index","best","worst"}), one per
judge batch, so a fan-out of judges can each write its own file.

    python scripts/assemble_rlaif_prefs.py --candidates data/corpora/rlaif_candidates.jsonl \\
        --verdicts /path/to/rlaif_verdicts --out data/corpora/rlaif_prefs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_records(candidates: str | Path) -> list[dict]:
    """Candidate records in file order; the 0-based line number is the record index."""
    return [json.loads(line) for line in Path(candidates).read_text().splitlines() if line.strip()]


def load_verdicts(paths: list[str | Path]) -> dict[int, dict]:
    """Merge verdict files into {index: verdict}. Each file is a JSON list of objects carrying
    an integer 'index'. On a duplicate index the later file wins. An unreadable file (e.g. a
    partial write from an interrupted judge) is skipped with a warning, not fatal — the resume
    re-judges those batches."""
    verdicts: dict[int, dict] = {}
    for p in paths:
        try:
            data = json.loads(Path(p).read_text())
        except (json.JSONDecodeError, OSError):
            print(f"warning: skipping unreadable verdict file {p}", file=sys.stderr)
            continue
        for v in data:
            verdicts[int(v["index"])] = v
    return verdicts


def build_pairs(records: list[dict], verdicts: dict[int, dict]) -> list[dict]:
    """chosen = candidates[best], rejected = candidates[worst]. A record is dropped when it has
    no verdict, best == worst, an index is out of range, or the two picks are identical text."""
    pairs: list[dict] = []
    for idx, rec in enumerate(records):
        v = verdicts.get(idx)
        if not v:
            continue
        best, worst = v.get("best"), v.get("worst")
        cands = rec["candidates"]
        if best is None or worst is None or best == worst:
            continue
        if not (0 <= best < len(cands) and 0 <= worst < len(cands)):
            continue
        if cands[best].strip() == cands[worst].strip():
            continue
        pairs.append({"prompt": rec["prompt"], "chosen": cands[best], "rejected": cands[worst]})
    return pairs


def write_prefs(pairs: list[dict], out: str | Path) -> int:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    return len(pairs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", default="data/corpora/rlaif_candidates.jsonl")
    ap.add_argument("--verdicts", required=True,
                    help="a verdict .json file, or a directory of them")
    ap.add_argument("--out", default="data/corpora/rlaif_prefs.jsonl")
    args = ap.parse_args()

    vpath = Path(args.verdicts)
    # Only the per-batch verdict files — not the judge's _schema.json or any .tmp left by a crash.
    verdict_files = sorted(vpath.glob("verdicts_*.json")) if vpath.is_dir() else [vpath]
    if not verdict_files:
        raise FileNotFoundError(f"no verdicts_*.json files under {args.verdicts}")

    records = load_records(args.candidates)
    verdicts = load_verdicts(verdict_files)
    pairs = build_pairs(records, verdicts)
    n = write_prefs(pairs, args.out)
    print(f"{len(records)} candidate records, {len(verdicts)} verdicts, {len(verdict_files)} "
          f"verdict file(s) -> wrote {n} preference pairs to {args.out}")


if __name__ == "__main__":
    main()
