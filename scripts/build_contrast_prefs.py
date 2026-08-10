"""Correctness-contrast IPO pairs from competitive problems (chosen=passing,
rejected=wrong-output; executor-labeled, human-authored — build-capability compliant).

    python scripts/build_contrast_prefs.py --out data/corpora/contrast_prefs.jsonl \\
        --limit-per-dataset 3000
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
    contrast_pairs,
    decontaminate,
    taco_problem,
)


def main() -> None:  # pragma: no cover - network + sandbox operational
    from datasets import load_dataset

    from microlab.evals.code.tasks import load_humaneval, load_mbpp

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/corpora/contrast_prefs.jsonl")
    ap.add_argument("--limit-per-dataset", type=int, default=3000)
    ap.add_argument("--max-cases", type=int, default=6)
    ap.add_argument("--max-solutions", type=int, default=8)
    ap.add_argument("--timeout-s", type=float, default=5.0)
    args = ap.parse_args()

    adapters = [
        ("json", "hf://datasets/codeparrot/apps/train.jsonl", apps_problem),
        ("deepmind/code_contests", None, codecontests_problem),
        ("parquet", "hf://datasets/BAAI/TACO/ALL/train-*.parquet", taco_problem),
    ]
    all_pairs, tally = [], {"problems": 0, "pairs": 0, "no_pair": 0}
    for fmt, data_files, adapt in adapters:
        it = load_dataset(fmt, data_files=data_files, split="train", streaming=True) \
            if data_files else load_dataset(fmt, split="train", streaming=True)
        probs, seen = [], 0
        for r in it:
            probs.append(adapt(r))
            seen += 1
            if args.limit_per_dataset and seen >= args.limit_per_dataset:
                break
        pairs, t = contrast_pairs(probs, max_cases=args.max_cases,
                                  max_solutions=args.max_solutions,
                                  timeout_s=args.timeout_s)
        all_pairs += pairs
        for k in tally:
            tally[k] += t[k]
        print(f"  {adapt.__name__}: {t}", flush=True)

    bench = [t.prompt + "\n" + t.test_program for t in (load_humaneval() + load_mbpp())]
    fp = benchmark_fingerprints(bench, n=10)
    mirrors = [{"instruction": p["prompt"], "context": "",
                "response": p["chosen"] + " " + p["rejected"]} for p in all_pairs]
    kept, removed = decontaminate(mirrors, fp, n=10)
    kept_prompts = {m["instruction"] for m in kept}
    all_pairs = [p for p in all_pairs if p["prompt"] in kept_prompts]

    print(f"contrast pairs: {tally} decontaminated_removed={removed} total={len(all_pairs)}")
    if not all_pairs:
        raise SystemExit("no contrast pairs — refusing to proceed (verify by count)")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")
    print(f"wrote {len(all_pairs)} pairs -> {out}")


if __name__ == "__main__":
    main()
