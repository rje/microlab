"""Generate the TEST-FREE MBPP sample bank (spec finding C1): prompts carry the task
description + bare function signature ONLY — never the gold asserts that mbpp_task's
standard chat instruction embeds. This bank is Unit 2's leakage-clean powered comparison;
its pass@k vs the standard bank's also measures test-conditioning.

    python scripts/gen_testfree_bank.py --policy runs/coder-1b-instruct-compliant \\
        --out evals/harness/mbpp-testfree-bank.jsonl --n 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def mbpp_signature(code: str, entry_point: str | None = None) -> str | None:
    """The bare `def ...:` signature line of the reference solution, else None.

    With entry_point: the def line whose function name matches it (4/257 MBPP references
    define a helper BEFORE the entry point — first-def would hand the model the wrong
    signature). Without: the first def line (legacy behavior)."""
    for line in (code or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("def ") and stripped.endswith(":"):
            if entry_point is None:
                return stripped
            name = stripped[len("def "):].split("(", 1)[0].strip()
            if name == entry_point:
                return stripped
    return None


def plan_resume(rows: list[dict], n: int) -> tuple[set[str], list[dict]]:
    """From existing bank rows: (complete_task_ids, compacted_rows). A task is COMPLETE
    only with exactly n sample rows; partial tasks' rows are dropped from compacted_rows
    (they regenerate deterministically — sampling is seeded). Exactly one _header row is
    kept (the first); extras from crash-resume are dropped."""
    header = next((r for r in rows if "_header" in r), None)
    counts: dict[str, int] = {}
    for r in rows:
        if "task_id" in r:
            counts[r["task_id"]] = counts.get(r["task_id"], 0) + 1
    complete = {t for t, c in counts.items() if c == n}
    compacted = [header] if header is not None else []
    compacted += [r for r in rows if r.get("task_id") in complete]
    return complete, compacted


def testfree_instruction(prompt: str, signature: str) -> str:
    """Description + signature, no tests. The generator must stay blind to gold asserts."""
    return (f"{prompt.strip()}\n\nWrite the complete Python function "
            f"`{signature}` Reply with the full function definition.")


def main() -> None:  # pragma: no cover - GPU + sandbox operational
    import torch
    from datasets import load_dataset

    from microlab.evals.code.executor import run_python
    from microlab.evals.code.tasks import assemble_program, mbpp_task
    from microlab.model.reference.checkpoint import load_variant_from_run
    from microlab.model.reference.sft import format_chat
    from microlab.tokenizer.fast import FastTokenizer
    from microlab.train.exec_reward import extract_solution, sample_solutions

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="runs/coder-1b-instruct-compliant")
    ap.add_argument("--out", default="evals/harness/mbpp-testfree-bank.jsonl")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout-s", type=float, default=8.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rows = list(load_dataset("google-research-datasets/mbpp", "sanitized", split="test"))
    if args.limit:
        rows = rows[:args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = ([json.loads(x) for x in out.read_text().splitlines() if x.strip()]
                if out.exists() else [])
    done, compacted = plan_resume(existing, args.n)
    if existing:
        partial = len({r["task_id"] for r in existing if "task_id" in r} - done)
        with out.open("w", encoding="utf-8") as f:
            for r in compacted:
                f.write(json.dumps(r) + "\n")
        print(f"resume: {len(done)} complete tasks kept, {partial} partial tasks reset")
    model, _ = load_variant_from_run(Path(args.policy), device=args.device)
    tok = FastTokenizer.load(str(Path(args.policy) / "tokenizer.json"))

    with out.open("a", encoding="utf-8") as f:
        if not any("_header" in r for r in compacted):
            f.write(json.dumps({"_header": {
                "run": args.policy, "dataset": "mbpp-testfree", "n": args.n,
                "temperature": args.temp, "top_k": args.top_k, "seed": args.seed,
                "note": "generation prompts contain NO gold asserts (spec C1)"}}) + "\n")
        skipped_sig = 0
        for i, row in enumerate(rows):
            task = mbpp_task(row)
            if task.task_id in done:
                continue
            sig = mbpp_signature(row.get("code", ""), entry_point=task.entry_point)
            if sig is None:
                skipped_sig += 1
                continue
            prompt, _ = format_chat(testfree_instruction(row["prompt"], sig), "")
            replies = sample_solutions(model, tok._tok, prompt, args.n,
                                       max_new=args.max_new, temp=args.temp,
                                       top_k=args.top_k, seed=args.seed,
                                       device=args.device)
            for s_idx, rep in enumerate(replies):
                sol = extract_solution(rep)
                ok = bool(sol.strip()) and run_python(
                    assemble_program(sol, task), timeout_s=args.timeout_s).passed
                f.write(json.dumps({"task_id": task.task_id, "sample": s_idx,
                                    "passed": ok, "solution": sol}) + "\n")
            f.flush()
            if (i + 1) % 20 == 0:
                print(f"  bank {i + 1}/{len(rows)}", flush=True)
    print(f"skipped (no signature): {skipped_sig}")

    bank = [json.loads(x) for x in out.read_text().splitlines()]
    tasks_written = {r["task_id"] for r in bank if "task_id" in r}
    if not tasks_written:
        raise SystemExit("empty bank — refusing to summarize (verify by count)")
    # summary via the shared delivered() logic
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(
        "eval_rerank", Path(__file__).resolve().parent / "eval_rerank.py")
    er = ilu.module_from_spec(spec)
    spec.loader.exec_module(er)
    rep = er.delivered(bank)
    Path(str(out) + ".summary.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(f"summary: {rep}")


if __name__ == "__main__":
    main()
