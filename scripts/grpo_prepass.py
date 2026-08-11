"""Policy pre-pass: measure v1's per-problem success on the GRPO pool, keep the
signal-bearing subset (0 < successes < k).

    python scripts/grpo_prepass.py --policy runs/coder-1b-instruct-compliant \\
        --pool data/corpora/grpo_pool.jsonl --k 8 \\
        --stats data/corpora/grpo_prepass_stats.jsonl \\
        --out data/corpora/grpo_pool_signal.jsonl

Progressive + resumable: stats append per problem; already-measured instructions are
skipped on rerun. The signal pool is (re)written whole from the stats at the end.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from microlab.model.reference.checkpoint import load_variant_from_run  # noqa: E402
from microlab.model.reference.sft import format_chat  # noqa: E402
from microlab.tokenizer.fast import FastTokenizer  # noqa: E402
from microlab.train.exec_reward import (  # noqa: E402
    extract_solution,
    io_reward,
    sample_solutions,
    signal_bearing,
)


def main() -> None:  # pragma: no cover - GPU + sandbox operational script
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="runs/coder-1b-instruct-compliant")
    ap.add_argument("--pool", default="data/corpora/grpo_pool.jsonl")
    ap.add_argument("--stats", default="data/corpora/grpo_prepass_stats.jsonl")
    ap.add_argument("--out", default="data/corpora/grpo_pool_signal.jsonl")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--timeout-s", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="pool rows (smoke)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    pool = [json.loads(x) for x in Path(args.pool).read_text().splitlines()]
    if args.limit:
        pool = pool[:args.limit]
    done = set()
    stats_path = Path(args.stats)
    if stats_path.exists():
        done = {json.loads(x)["instruction"] for x in stats_path.read_text().splitlines()}

    model, _ = load_variant_from_run(Path(args.policy), device=args.device)
    tok = FastTokenizer.load(str(Path(args.policy) / "tokenizer.json"))

    with stats_path.open("a", encoding="utf-8") as f:
        for i, row in enumerate(pool):
            if row["instruction"] in done:
                continue
            prompt, _ = format_chat(row["instruction"], "")
            # top_k=None: the pre-pass certifies problems FOR GRPO, so it must sample from
            # the SAME distribution training rollouts use (run_grpo's sample_group applies
            # no top-k). A more generous pre-pass (top-40) optimistically certifies problems
            # that starve under real rollouts. Keep --max-new equal to train_grpo's too.
            replies = sample_solutions(model, tok._tok, prompt, args.k,
                                       max_new=args.max_new, top_k=None, seed=args.seed,
                                       device=args.device)
            rewards = [io_reward(extract_solution(r), row["io"], timeout_s=args.timeout_s)
                       for r in replies]
            successes = sum(1 for r in rewards if r == 1.0)
            f.write(json.dumps({"instruction": row["instruction"],
                                "successes": successes, "k": args.k,
                                "rewards": rewards}) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"  prepass {i + 1}/{len(pool)}", flush=True)

    stats = {json.loads(x)["instruction"]: json.loads(x)
             for x in stats_path.read_text().splitlines()}
    signal = [r for r in pool
              if r["instruction"] in stats
              and signal_bearing(stats[r["instruction"]]["successes"],
                                 stats[r["instruction"]]["k"])]
    print(f"prepass: pool={len(pool)} measured={len(stats)} signal={len(signal)}")
    if not signal:
        raise SystemExit("no signal-bearing problems — GRPO would starve (verify by count)")
    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        for r in signal:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(signal)} signal rows -> {out}")


if __name__ == "__main__":
    main()
