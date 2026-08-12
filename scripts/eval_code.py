"""HumanEval / MBPP execution eval for microlab runs.

    python scripts/eval_code.py --run runs/1b-4k-chat --dataset humaneval \\
        --out evals/code/1b-4k-chat-humaneval.jsonl

Generates a completion per task (greedy pass@1 by default; --n N --temperature T for
sampled pass@k), assembles candidate + reference tests into one program, and executes it
under the sandboxed executor (timeout, memory cap, tmpdir, no-network where the kernel
allows). A task passes iff the program exits 0.

Modes: --mode base feeds the completion-style prompt verbatim (HumanEval signature +
docstring; MBPP docstring + first assert) and truncates the continuation at the first
new top-level construct; --mode chat wraps the task instruction in the trained SFT
template and mines the reply for a code block. --mode auto reads serve_config.json in
the run dir and raises when there is none — pass the mode explicitly for base runs.

Results append to --out (JSONL, one line per task sample, resumable: finished task ids
are skipped on rerun; the header line pins the config and a mismatch raises). The
summary (pass@1, and pass@10 when --n >= 10) is written next to it as .summary.json.

MultiPL-E JS/TS: not implemented — the task interface is language-agnostic (see
microlab.evals.code.tasks.load_tasks) but execution needs a sandboxed node runtime the
executor does not provide yet."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from microlab.evals.code.executor import run_python  # noqa: E402
from microlab.evals.code.gen import generate_until  # noqa: E402
from microlab.evals.code.prompts import (  # noqa: E402
    BASE_STOPS,
    CHAT_STOPS,
    base_solution,
    chat_prompt,
    chat_solution,
)
from microlab.evals.code.tasks import assemble_program, load_tasks  # noqa: E402
from microlab.evals.reference.metrics import pass_at_k  # noqa: E402
from microlab.infer.batched import generate_batch  # noqa: E402
from microlab.model.reference.checkpoint import load_variant_from_run  # noqa: E402
from microlab.tokenizer.fast import FastTokenizer  # noqa: E402


def resolve_mode(mode: str, run_dir: Path) -> str:
    """'auto' resolves through serve_config.json; anything else passes through."""
    if mode != "auto":
        return mode
    cfg_path = run_dir / "serve_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"--mode auto but {cfg_path} does not exist; pass --mode base or --mode chat"
        )
    return json.loads(cfg_path.read_text(encoding="utf-8"))["mode"]


def read_resume(out_path: Path, header: dict) -> set[str]:
    """Sample keys already in the JSONL. The stored header must match ours exactly —
    silently mixing configs would corrupt the numbers."""
    if not out_path.exists():
        return set()
    lines = [json.loads(x) for x in
             out_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not lines:
        return set()
    if "_header" not in lines[0] or lines[0]["_header"] != header:
        raise ValueError(
            f"{out_path} exists with a different config header; "
            f"point --out somewhere fresh (have {lines[0].get('_header')}, want {header})"
        )
    return {f"{r['task_id']}#{r['sample']}" for r in lines[1:]}


def summarize(records: list[dict], n: int) -> dict:
    """pass@1 (+pass@10 when n >= 10) from per-sample pass/fail records."""
    by_task: dict[str, int] = {}
    counts: dict[str, int] = {}
    for r in records:
        by_task[r["task_id"]] = by_task.get(r["task_id"], 0) + (1 if r["passed"] else 0)
        counts[r["task_id"]] = counts.get(r["task_id"], 0) + 1
    bad = {t: c for t, c in counts.items() if c != n}
    if bad:
        raise ValueError(f"tasks with sample count != n={n}: {bad}")
    ks = [k for k in (1, 10) if k <= n]
    out = {"n_tasks": len(by_task), "n_samples_per_task": n}
    for k in ks:
        out[f"pass@{k}"] = sum(pass_at_k(n, c, k) for c in by_task.values()) / len(by_task)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--dataset", default="humaneval",
                    choices=["humaneval", "mbpp", "multipl-js", "multipl-ts"])
    ap.add_argument("--mode", default="auto", choices=["auto", "base", "chat"])
    ap.add_argument("--out", required=True, type=Path,
                    help="results JSONL (appended progressively; resumable)")
    ap.add_argument("--n", type=int, default=1, help="samples per task (pass@k)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--limit", type=int, default=None, help="first N tasks only")
    ap.add_argument("--timeout-s", type=float, default=10.0)
    ap.add_argument("--memory-mb", type=int, default=512)
    ap.add_argument("--require-netns", action="store_true",
                    help="refuse to execute without network-namespace isolation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--engine", default="sequential", choices=["sequential", "batched"],
                    help="batched: all n samples of a task decoded as one batch "
                         "(~4x wall-clock; a DIFFERENT deterministic sampling stream "
                         "than sequential — the header records it, so files never mix)")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    args = ap.parse_args()

    if args.n > 1 and args.temperature <= 0.0:
        raise ValueError("--n > 1 needs --temperature > 0 (greedy samples are identical)")
    if args.n == 1 and args.temperature != 0.0:
        raise ValueError("pass@1 baseline is greedy; drop --temperature or set --n")

    mode = resolve_mode(args.mode, args.run)
    tasks = load_tasks(args.dataset)
    if args.limit is not None:
        tasks = tasks[: args.limit]

    header = {"run": str(args.run), "dataset": args.dataset, "mode": mode, "n": args.n,
              "temperature": args.temperature, "max_new": args.max_new,
              "seed": args.seed, "limit": args.limit,
              "engine": args.engine, "dtype": args.dtype}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = read_resume(args.out, header)
    if not done and (not args.out.exists() or not args.out.read_text().strip()):
        with args.out.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"_header": header}) + "\n")

    model, step = load_variant_from_run(args.run, device=args.device)
    if args.dtype == "bf16":
        model = model.to(torch.bfloat16)
    tok = FastTokenizer.load(str(args.run / "tokenizer.json"))
    print(f"run={args.run} step={step} mode={mode} dataset={args.dataset} "
          f"tasks={len(tasks)} n={args.n} (resuming past {len(done)} samples)")

    t0 = time.time()
    for ti, task in enumerate(tasks):
        if mode == "chat":
            prompt = chat_prompt(task.instruction)
            stops = CHAT_STOPS
        else:
            prompt = task.prompt
            stops = BASE_STOPS[args.dataset]
        if args.engine == "batched":
            # All n samples in one batch. Deterministic for (seed, n): a crash-resumed
            # partial task regenerates identical rows, so already-written keys are
            # simply skipped below rather than duplicated.
            if all(f"{task.task_id}#{s}" in done for s in range(args.n)):
                continue
            gen_kw = {}
            if args.temperature > 0:
                gen_kw["generator"] = torch.Generator(device=args.device).manual_seed(
                    args.seed * 1_000_003 + ti * 1009)
            completions = generate_batch(
                model, tok, tok.encode(prompt), n=args.n, max_new=args.max_new,
                stops=stops, device=args.device, temperature=args.temperature,
                top_k=args.top_k, **gen_kw)
        else:
            completions = None
        for s in range(args.n):
            key = f"{task.task_id}#{s}"
            if key in done:
                continue
            if completions is not None:
                completion = completions[s]
            else:
                gen_kw = {}
                if args.temperature > 0:
                    gen_kw["generator"] = torch.Generator(
                        device=args.device).manual_seed(
                        args.seed * 1_000_003 + ti * 1009 + s)
                completion = generate_until(
                    model, tok, tok.encode(prompt), max_new=args.max_new, stops=stops,
                    device=args.device, temperature=args.temperature, top_k=args.top_k,
                    **gen_kw)
            solution = (chat_solution(completion) if mode == "chat"
                        else base_solution(args.dataset, task, completion))
            res = run_python(
                assemble_program(solution, task), timeout_s=args.timeout_s,
                memory_mb=args.memory_mb, require_netns=args.require_netns)
            rec = {"task_id": task.task_id, "sample": s, "passed": res.passed,
                   "exit_code": res.exit_code, "timed_out": res.timed_out,
                   "network_isolation": res.network_isolation, "solution": solution,
                   "stderr_tail": res.stderr[-800:]}
            with args.out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            mark = "PASS" if res.passed else ("TIME" if res.timed_out else "fail")
            print(f"[{ti + 1}/{len(tasks)}] {key}: {mark} [{time.time() - t0:.0f}s]",
                  flush=True)

    records = [json.loads(x) for x in
               args.out.read_text(encoding="utf-8").splitlines()[1:] if x.strip()]
    summary = {"header": header, "ckpt_step": step, **summarize(records, args.n)}
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out} and {summary_path} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
