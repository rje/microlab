"""Delivered correctness under executor reranking, from an eval_code --n K output.

With tests available, best-of-k + executor rerank delivers a task iff ANY sample passes —
so delivered_rate == pass@any-of-k, computed from the per-sample rows eval_code already
writes. pass@1_first_sample is the unbiased single-draw baseline for the same run.

    python scripts/eval_rerank.py evals/instruct/<run>-humaneval-sampled.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def delivered(rows: list[dict]) -> dict:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if "_header" in r:
            continue
        by_task[r["task_id"]].append(r)
    n = len(by_task)
    if n == 0:
        raise ValueError("no task rows — wrong file? (verify by count)")
    k = max(len(v) for v in by_task.values())
    correct = sum(1 for v in by_task.values() if any(s["passed"] for s in v))
    first = sum(1 for v in by_task.values()
                if any(s["passed"] and s.get("sample") == 0 for s in v))
    return {"n_tasks": n, "k": k, "delivered_correct": correct,
            "delivered_rate": correct / n, "pass@1_first_sample": first / n}


def main() -> None:  # pragma: no cover - thin IO wrapper
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rows = [json.loads(x) for x in args.jsonl.read_text().splitlines()]
    rep = delivered(rows)
    print(json.dumps(rep, indent=2))
    if args.out:
        args.out.write_text(json.dumps(rep, indent=2) + "\n")


if __name__ == "__main__":
    main()
