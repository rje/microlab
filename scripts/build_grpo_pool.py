"""Build the GRPO prompt pool: competitive problems + capped I/O cases, decontaminated.

    python scripts/build_grpo_pool.py --out data/corpora/grpo_pool.jsonl \\
        --limit-per-dataset 4000 --max-cases 6

Emits {"instruction", "io": [{"input","output"}, ...]} per line. The policy pre-pass
(grpo_prepass.py) filters this to the signal-bearing subset before training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microlab.data.code_sft import (  # noqa: E402
    apps_problem,
    benchmark_fingerprints,
    codecontests_problem,
    decontaminate,
    taco_problem,
)


def pool_row(problem: dict, max_cases: int = 6) -> dict | None:
    """Normalized problem -> pool row, or None when unusable (no statement / no cases)."""
    statement = (problem.get("statement") or "").strip()
    cases = (problem.get("io") or [])[:max_cases]
    if not statement or not cases:
        return None
    return {"instruction": statement, "io": cases}


def main() -> None:  # pragma: no cover - network + IO
    from datasets import load_dataset

    from microlab.evals.code.tasks import load_humaneval, load_mbpp

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/corpora/grpo_pool.jsonl")
    ap.add_argument("--limit-per-dataset", type=int, default=4000)
    ap.add_argument("--max-cases", type=int, default=6)
    args = ap.parse_args()

    adapters = [
        ("json", "hf://datasets/codeparrot/apps/train.jsonl", apps_problem),
        ("deepmind/code_contests", None, codecontests_problem),
        ("parquet", "hf://datasets/BAAI/TACO/ALL/train-*.parquet", taco_problem),
    ]
    rows, per_source = [], {}
    for fmt, data_files, adapt in adapters:
        it = load_dataset(fmt, data_files=data_files, split="train", streaming=True) \
            if data_files else load_dataset(fmt, split="train", streaming=True)
        n0, seen = len(rows), 0
        for r in it:
            seen += 1
            row = pool_row(adapt(r), max_cases=args.max_cases)
            if row:
                rows.append(row)
            if args.limit_per_dataset and seen >= args.limit_per_dataset:
                break
        per_source[adapt.__name__] = len(rows) - n0

    bench = [t.prompt + "\n" + t.test_program for t in (load_humaneval() + load_mbpp())]
    fp = benchmark_fingerprints(bench, n=10)
    mirrors = [{"instruction": r["instruction"], "context": "", "response": ""} for r in rows]
    kept_mirrors, removed = decontaminate(mirrors, fp, n=10)
    kept_ins = {m["instruction"] for m in kept_mirrors}
    rows = [r for r in rows if r["instruction"] in kept_ins]

    print(f"pool: {per_source} decontaminated_removed={removed} total={len(rows)}")
    if not rows:
        raise SystemExit("empty pool — refusing to proceed (verify by count)")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} pool rows -> {out}")


if __name__ == "__main__":
    main()
