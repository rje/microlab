"""Unit 2 runner: no-tests selection methods vs the oracle, per bank.

Selection is blind to hidden tests; hidden tests only SCORE the selected candidate.
The beat-the-oracle check is a leakage falsifier IN CODE: no method may pass where the
oracle (any-of-k on hidden tests) finds no passing candidate.

    python scripts/eval_selection.py --bank evals/harness/mbpp-testfree-bank.jsonl \\
        --inputs evals/harness/synth_inputs.jsonl --dataset mbpp \\
        --out evals/harness/selection-mbpp-testfree.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microlab.infer.selection import (  # noqa: E402
    behavior_clusters,
    first_sample,
    pick_from_cluster,
    select_by_self_tests,
    text_plurality,
)


def bank_by_task(rows: list[dict]) -> dict[str, list[dict]]:
    """Group bank rows by task, ordered by sample index. Raises on empty (verify by count)."""
    by: dict[str, list[dict]] = {}
    for r in rows:
        if "_header" in r:
            continue
        by.setdefault(r["task_id"], []).append(r)
    if not by:
        raise ValueError("no task rows in bank — wrong file? (verify by count)")
    for v in by.values():
        v.sort(key=lambda r: r["sample"])
    return by


def run_methods(candidates: list[str], passed: list[bool],
                signatures: list[tuple] | None, assert_counts: list[int] | None,
                seed: int = 0, oracle_override: list[bool] | None = None) -> dict:
    """Selected index per method; None where the method's inputs are unavailable.
    oracle_override exists only for the leakage self-test."""
    oracle_passed = oracle_override if oracle_override is not None else passed
    out: dict[str, int | None] = {"first": first_sample(),
                                  "text_plurality": text_plurality(candidates)}
    if signatures is not None:
        clusters = behavior_clusters(signatures)
        out["cluster_shortest"] = pick_from_cluster(clusters[0], candidates, "shortest")
        out["cluster_random"] = pick_from_cluster(clusters[0], candidates, "random", seed)
    else:
        out["cluster_shortest"] = out["cluster_random"] = None
    out["self_tests"] = (select_by_self_tests(assert_counts)
                         if assert_counts is not None else None)
    out["oracle"] = next((i for i, p in enumerate(oracle_passed) if p), None)
    if out["oracle"] is None:
        for m, idx in out.items():
            if m != "oracle" and idx is not None and passed[idx]:
                raise RuntimeError(
                    f"method {m!r} selected a passing candidate where the oracle found "
                    f"none — hidden tests leaked into selection (falsifier)")
    return out


def summarize(per_task: dict[str, dict[str, bool | None]]) -> dict:
    """per_task: task -> method -> did-the-selected-candidate-pass (None = no coverage).
    Reports counts, coverage, the recoverable gap, and the power verdict."""
    methods = sorted({m for d in per_task.values() for m in d})
    rep: dict = {"n_tasks": len(per_task), "methods": {}}
    for m in methods:
        vals = [d.get(m) for d in per_task.values()]
        rep["methods"][m] = {"passes": sum(1 for v in vals if v),
                             "coverage": sum(1 for v in vals if v is not None)}
    first_p = rep["methods"].get("first", {}).get("passes", 0)
    oracle_p = rep["methods"].get("oracle", {}).get("passes", 0)
    rep["recoverable_gap"] = oracle_p - first_p
    rep["underpowered"] = rep["recoverable_gap"] < 15
    return rep


def main() -> None:  # pragma: no cover - sandbox operational
    from microlab.infer.behavior import signatures_for

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--inputs", default=None,
                    help="jsonl {task_id, entry_point, exprs:[...]} of SYNTHESIZED inputs")
    ap.add_argument("--docstring-inputs", default=None,
                    help="jsonl of docstring-extracted inputs — reported SEPARATELY")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout-s", type=float, default=3.0)
    args = ap.parse_args()

    rows = [json.loads(x) for x in Path(args.bank).read_text().splitlines()]
    banks = bank_by_task(rows)

    def load_inputs(path):
        if not path or not Path(path).exists():
            return {}
        return {json.loads(x)["task_id"]: json.loads(x)
                for x in Path(path).read_text().splitlines()}

    input_sets = {"synth": load_inputs(args.inputs),
                  "docstring": load_inputs(args.docstring_inputs)}
    report: dict = {}
    for src, inputs in input_sets.items():
        if not inputs and src == "docstring":
            continue
        per_task: dict[str, dict] = {}
        for task_id, samples in banks.items():
            cands = [s["solution"] for s in samples]
            passed = [bool(s["passed"]) for s in samples]
            sigs = None
            meta = inputs.get(task_id)
            if meta and meta.get("exprs"):
                sigs = [signatures_for(c, meta["entry_point"], meta["exprs"][:4],
                                       timeout_s=args.timeout_s) for c in cands]
            sel = run_methods(cands, passed, sigs, None, seed=args.seed)
            per_task[task_id] = {m: (passed[i] if i is not None else None)
                                 for m, i in sel.items()}
        report[src] = summarize(per_task)
        print(f"[{src}] {json.dumps(report[src], indent=1)}")

    if not report:
        raise SystemExit("no input source produced a report (verify by count)")
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
