"""Build the COMPLIANT (no-distill) code-instruction SFT mix for coder-1b-instruct (arm A).

Sources (all human-authored or executor-verified-human):
  - CommitPackFT (bigcode/commitpackft)         commit message -> new file   [Python-first]
  - MBPP sanitized train/validation/prompt      problem text -> reference code
  - OASST1 code threads (OpenAssistant/oasst1)   multi-turn, code-bearing
  - APPS / CodeContests / TACO                   statement -> executor-VERIFIED solution

    python scripts/build_code_sft.py --out data/corpora/code_sft_compliant.jsonl

build_compliant_mix() is pure over already-loaded rows so it is unit-tested without the
network; main() streams the real sources, verifies competitive solutions in the sandbox,
decontaminates against the eval benchmarks, and prints a count/token/verify report.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microlab.data.code_sft import (  # noqa: E402
    apps_problem,
    benchmark_fingerprints,
    codecontests_problem,
    decontaminate,
    normalize_commitpack,
    normalize_mbpp_train,
    oasst_code_convs,
    taco_problem,
    total_supervised_tokens,
    verified_competitive_rows,
)

PY_LANGS = {"python"}
OUT_DEFAULT = "data/corpora/code_sft_compliant.jsonl"


def build_compliant_mix(sources: dict, tok, seed: int = 0) -> tuple[list[dict], dict]:
    """Merge already-normalized source row-lists, seed-shuffle, and report counts + supervised
    tokens. `sources` keys: commitpack, mbpp_train, oasst, competitive (each a list of rows)."""
    order = ["commitpack", "mbpp_train", "oasst", "competitive"]
    mix: list[dict] = []
    counts = {}
    for key in order:
        rows = sources.get(key, [])
        counts[key] = len(rows)
        mix += rows
    random.Random(seed).shuffle(mix)
    counts["total"] = len(mix)
    return mix, {"counts": counts, "supervised_tokens": total_supervised_tokens(mix, tok)}


def _load_sources(args, tok) -> dict:  # pragma: no cover - network/HF streaming
    from datasets import load_dataset
    lim = args.limit_per_source

    def cap(rows):
        return rows[:lim] if lim else rows

    commit = []
    for r in load_dataset("bigcode/commitpackft", "python", split="train", streaming=True):
        n = normalize_commitpack(r, PY_LANGS)
        if n:
            commit.append(n)
        if lim and len(commit) >= lim:
            break

    mbpp = []
    for split in ("train", "validation", "prompt"):
        for r in load_dataset("google-research-datasets/mbpp", "sanitized", split=split):
            n = normalize_mbpp_train(r)
            if n:
                mbpp.append(n)
    mbpp = cap(mbpp)

    oasst_msgs = list(load_dataset("OpenAssistant/oasst1", split="train"))
    oasst = cap(oasst_code_convs(oasst_msgs))

    comp_rows, tally = [], {"problems": 0, "verified": 0, "no_passing_solution": 0}
    adapters = [("codeparrot/apps", None, "test", apps_problem),
                ("deepmind/code_contests", None, "train", codecontests_problem),
                ("BAAI/TACO", None, "train", taco_problem)]
    for name, cfg, split, adapt in adapters:
        probs = []
        it = load_dataset(name, cfg, split=split, streaming=True) if cfg else \
            load_dataset(name, split=split, streaming=True)
        for r in it:
            probs.append(adapt(r))
            if lim and len(probs) >= lim:
                break
        rows, t = verified_competitive_rows(probs, max_per_problem=args.max_per_problem,
                                            timeout_s=args.timeout_s)
        comp_rows += rows
        for k in tally:
            tally[k] += t[k]
    print(f"competitive verify tally: {tally}", flush=True)
    return {"commitpack": commit, "mbpp_train": mbpp, "oasst": oasst, "competitive": comp_rows}


def main() -> None:  # pragma: no cover - network + IO
    from microlab.evals.code.tasks import load_humaneval, load_mbpp
    from microlab.tokenizer.fast import FastTokenizer

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--tokenizer", default="runs/coder-1b-step40000/tokenizer.json")
    ap.add_argument("--limit-per-source", type=int, default=None)
    ap.add_argument("--max-per-problem", type=int, default=1)
    ap.add_argument("--timeout-s", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heldout-out", default="data/corpora/code_sft_heldout.jsonl",
                    help="pairwise-judge held-out slice, excluded from BOTH training arms")
    ap.add_argument("--heldout-n", type=int, default=200,
                    help="number of rows to reserve for the held-out pairwise set")
    args = ap.parse_args()

    tok = FastTokenizer.load(args.tokenizer)
    sources = _load_sources(args, tok)
    mix, report = build_compliant_mix(sources, tok, seed=args.seed)

    # Decontaminate against the eval benchmarks (prompts + canonical solutions).
    bench = [t.prompt + "\n" + t.test_program for t in (load_humaneval() + load_mbpp())]
    fp = benchmark_fingerprints(bench, n=10)
    mix, removed = decontaminate(mix, fp, n=10)
    report["decontaminated_removed"] = removed

    # Reserve a held-out slice for the pairwise judge so it isn't run on arm A's training
    # data (the token-match target for arm B is computed from args.out, i.e. `train`, so
    # this correctly excludes the held-out rows from that budget too).
    heldout = mix[:args.heldout_n]
    train = mix[args.heldout_n:]
    report["counts"]["total"] = len(train)
    report["heldout_rows"] = len(heldout)

    if not train:
        raise SystemExit("empty mix — refusing to proceed (verify by count)")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")

    heldout_out = Path(args.heldout_out)
    heldout_out.parent.mkdir(parents=True, exist_ok=True)
    with heldout_out.open("w", encoding="utf-8") as f:
        for r in heldout:
            f.write(json.dumps(r) + "\n")

    print(f"report: {json.dumps(report)}")
    print(f"wrote {len(train)} rows -> {out}")
    print(f"wrote {len(heldout)} held-out rows -> {heldout_out}")


if __name__ == "__main__":
    main()
