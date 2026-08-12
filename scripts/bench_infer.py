"""Generation throughput benchmark for microlab runs — times GENERATION ONLY (no
executor), so engine/dtype changes can be compared on identical workloads.

    python scripts/bench_infer.py --run runs/coder-1b-instruct-compliant \\
        --dataset mbpp --limit 6 --n 10 --temperature 0.7 --top-k 40

Reports per-task and total wall-clock, generated tokens, tok/s, and peak CUDA memory.
Writes a JSON record to --out (appended, one line per invocation) so ladder rungs
accumulate in one file. The workload (tasks, prompts, sampling config) is pinned by the
args; the engine under test is picked with --engine/--dtype."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from microlab.evals.code.gen import generate_until  # noqa: E402
from microlab.evals.code.prompts import CHAT_STOPS, chat_prompt  # noqa: E402
from microlab.evals.code.tasks import load_tasks  # noqa: E402
from microlab.model.reference.checkpoint import load_variant_from_run  # noqa: E402
from microlab.tokenizer.fast import FastTokenizer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--dataset", default="mbpp", choices=["humaneval", "mbpp"])
    ap.add_argument("--limit", type=int, default=6, help="first N tasks")
    ap.add_argument("--n", type=int, default=10, help="samples per task")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--engine", default="sequential", choices=["sequential", "batched"])
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--out", type=Path, default=Path("evals/bench/infer_bench.jsonl"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tasks = load_tasks(args.dataset)[: args.limit]
    model, step = load_variant_from_run(args.run, device=args.device)
    if args.dtype == "bf16":
        model = model.to(torch.bfloat16)
    tok = FastTokenizer.load(str(args.run / "tokenizer.json"))

    if args.engine == "batched":
        from microlab.infer.batched import generate_batch

    torch.cuda.reset_peak_memory_stats()
    per_task = []
    total_tokens = 0
    digest = hashlib.sha256()
    t_all = time.time()
    for ti, task in enumerate(tasks):
        prompt_ids = tok.encode(chat_prompt(task.instruction))
        t0 = time.time()
        if args.engine == "sequential":
            outs = []
            for s in range(args.n):
                gen = torch.Generator(device=args.device).manual_seed(
                    args.seed * 1_000_003 + ti * 1009 + s)
                outs.append(generate_until(
                    model, tok, prompt_ids, max_new=args.max_new, stops=CHAT_STOPS,
                    device=args.device, temperature=args.temperature,
                    top_k=args.top_k, generator=gen))
        else:
            gen = torch.Generator(device=args.device).manual_seed(
                args.seed * 1_000_003 + ti * 1009)
            outs = generate_batch(
                model, tok, prompt_ids, n=args.n, max_new=args.max_new,
                stops=CHAT_STOPS, device=args.device, temperature=args.temperature,
                top_k=args.top_k, generator=gen)
        dt = time.time() - t0
        toks = sum(len(tok.encode(o)) for o in outs)
        total_tokens += toks
        for o in outs:
            digest.update(o.encode())
            digest.update(b"\x00")
        per_task.append({"task_id": task.task_id, "wall_s": round(dt, 2),
                         "gen_tokens": toks, "tok_s": round(toks / dt, 1)})
        print(f"[{ti + 1}/{len(tasks)}] {task.task_id}: {dt:.1f}s "
              f"{toks} tok ({toks / dt:.1f} tok/s)", flush=True)

    wall = time.time() - t_all
    rec = {
        "engine": args.engine, "dtype": args.dtype, "run": str(args.run), "step": step,
        "dataset": args.dataset, "limit": args.limit, "n": args.n,
        "temperature": args.temperature, "top_k": args.top_k, "max_new": args.max_new,
        "seed": args.seed, "wall_s": round(wall, 2), "gen_tokens": total_tokens,
        "tok_s": round(total_tokens / wall, 1),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "output_sha": digest.hexdigest()[:16], "per_task": per_task,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "per_task"}, indent=2))


if __name__ == "__main__":
    main()
