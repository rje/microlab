"""Self-generated SFT tranche: the policy samples k solutions per pool problem; the
executor keeps full passers (SelfCodeAlign-style, own-model + executor labels — compliant).

    python scripts/build_selfgen_sft.py --policy runs/<best> \\
        --pool data/corpora/grpo_pool.jsonl --k 8 \\
        --work data/corpora/selfgen_work.jsonl --out data/corpora/selfgen_sft.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def selfgen_row(instruction: str, solutions_with_rewards: list[tuple[str, float]]) -> dict | None:
    """Shortest full-pass (reward == 1.0) solution -> SFT row; None when nothing fully
    passes (a partial pass must NOT become training signal)."""
    full = sorted((s for s, r in solutions_with_rewards if r == 1.0 and s.strip()), key=len)
    if not full:
        return None
    return {"instruction": instruction, "context": "", "response": full[0]}


def main() -> None:  # pragma: no cover - GPU + sandbox operational
    import torch

    from microlab.data.code_sft import benchmark_fingerprints, decontaminate
    from microlab.evals.code.tasks import load_humaneval, load_mbpp
    from microlab.model.reference.checkpoint import load_variant_from_run
    from microlab.model.reference.sft import format_chat
    from microlab.tokenizer.fast import FastTokenizer
    from microlab.train.exec_reward import extract_solution, io_reward, sample_solutions

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--pool", default="data/corpora/grpo_pool.jsonl")
    ap.add_argument("--work", default="data/corpora/selfgen_work.jsonl")
    ap.add_argument("--out", default="data/corpora/selfgen_sft.jsonl")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--timeout-s", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    pool = [json.loads(x) for x in Path(args.pool).read_text().splitlines()]
    if args.limit:
        pool = pool[:args.limit]
    work = Path(args.work)
    done = {json.loads(x)["instruction"] for x in work.read_text().splitlines()} \
        if work.exists() else set()

    model, _ = load_variant_from_run(Path(args.policy), device=args.device)
    tok = FastTokenizer.load(str(Path(args.policy) / "tokenizer.json"))

    with work.open("a", encoding="utf-8") as f:
        for i, row in enumerate(pool):
            if row["instruction"] in done:
                continue
            prompt, _ = format_chat(row["instruction"], "")
            replies = sample_solutions(model, tok._tok, prompt, args.k,
                                       max_new=args.max_new, seed=args.seed,
                                       device=args.device)
            swr = [(extract_solution(r),
                    io_reward(extract_solution(r), row["io"], timeout_s=args.timeout_s))
                   for r in replies]
            out_row = selfgen_row(row["instruction"], swr)
            f.write(json.dumps({"instruction": row["instruction"], "row": out_row}) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"  selfgen {i + 1}/{len(pool)}", flush=True)

    rows, seen_resp = [], set()
    for line in work.read_text().splitlines():
        d = json.loads(line)
        r = d.get("row")
        if r and r["response"] not in seen_resp:
            seen_resp.add(r["response"])
            rows.append(r)
    bench = [t.prompt + "\n" + t.test_program for t in (load_humaneval() + load_mbpp())]
    rows, removed = decontaminate(rows, benchmark_fingerprints(bench, n=10), n=10)
    print(f"selfgen: pool={len(pool)} kept={len(rows)} decontaminated_removed={removed}")
    if not rows:
        raise SystemExit("no self-gen rows — refusing to proceed (verify by count)")
    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
