"""Best-of-k with caller-provided tests (harness deliverable 3a, tests-required).

    python scripts/generate_best_of.py --instruction "Write add(a,b)" \\
        --asserts-file my_asserts.py --k 8

Samples k candidates from the frozen v1, executes each against the asserts, prints the
first passer. Exit 0 = a passing solution was printed; exit 3 = NONE passed (the first
sample is printed anyway, with a stderr warning) — callers must check the exit code.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def pick_best(solutions: list[str], results: list[bool]) -> int | None:
    """First index whose result is True; None when nothing passed."""
    for i, ok in enumerate(results):
        if ok:
            return i
    return None


def main() -> int:  # pragma: no cover - GPU + sandbox operational
    import torch

    from microlab.evals.code.executor import run_python
    from microlab.evals.code.prompts import CHAT_STOPS
    from microlab.infer.batched import generate_batch
    from microlab.model.reference.checkpoint import load_variant_from_run
    from microlab.model.reference.sft import format_chat
    from microlab.tokenizer.fast import FastTokenizer
    from microlab.train.exec_reward import extract_solution

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="runs/coder-1b-instruct-compliant")
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--asserts-file", required=True, type=Path,
                    help="python file of asserts run AFTER the candidate definition")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout-s", type=float, default=8.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "bf16"],
                    help="bf16 default: parity-gated vs fp32 (greedy MBPP 36/257 "
                         "identical; delivered slice in band) at half the memory")
    args = ap.parse_args()

    if args.k < 1:
        ap.error(f"--k must be >= 1 (got {args.k})")
    if args.temp == 0.0 and args.k > 1:
        ap.error(f"--temp 0 with --k {args.k} would produce {args.k} identical "
                 "greedy candidates; use --k 1 or a positive --temp")

    asserts = args.asserts_file.read_text()
    model, _ = load_variant_from_run(Path(args.policy), device=args.device)
    if args.dtype == "bf16":
        model = model.to(torch.bfloat16)
    tok = FastTokenizer.load(str(Path(args.policy) / "tokenizer.json"))
    prompt, _ = format_chat(args.instruction, "")
    gen = (torch.Generator(device=args.device).manual_seed(args.seed)
           if args.temp > 0 else None)
    replies = generate_batch(model, tok, tok.encode(prompt), n=args.k,
                             max_new=args.max_new, stops=CHAT_STOPS,
                             device=args.device, temperature=args.temp,
                             top_k=args.top_k, generator=gen)
    sols = [extract_solution(r) for r in replies]
    results = [bool(s.strip()) and run_python(s + "\n\n" + asserts,
                                              timeout_s=args.timeout_s).passed
               for s in sols]
    best = pick_best(sols, results)
    if best is None:
        print(sols[0])
        print(f"WARNING: none of {args.k} candidates passed the provided asserts",
              file=sys.stderr)
        return 3
    print(sols[best])
    print(f"# passed provided asserts (candidate {best + 1}/{args.k})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
