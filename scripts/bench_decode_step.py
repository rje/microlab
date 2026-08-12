"""Decode-step microbench: per-step latency vs batch size, plus the torch.compile
probes that justified rejecting compile in the efficiency pass.

    python scripts/bench_decode_step.py --run runs/coder-1b-instruct-compliant

Appends one JSON record to --out. The B-sweep (fixed-shape decode after a 600-token
prefill) is what shows the step is launch-bound (flat in B); --with-compile re-runs the
two compile probes (slow: full-model compile recompiles every step by design of the
growing-KV cache protocol — that failure IS the measurement)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from microlab.infer.reference.kv_cache import build_cache  # noqa: E402
from microlab.model.reference.checkpoint import load_variant_from_run  # noqa: E402


def step_ms(model, B: int, *, prefill: int = 600, iters: int = 30,
            device: str = "cuda") -> float:
    dtype = next(model.parameters()).dtype
    cache = build_cache(model, B, device, dtype=dtype)
    idx = torch.randint(100, 5000, (B, prefill), device=device)
    with torch.no_grad():
        logits, _ = model(idx, kv_cache=cache)
        nxt = logits[:, -1:].argmax(-1)
        for _ in range(5):
            logits, _ = model(nxt, kv_cache=cache)
            nxt = logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            logits, _ = model(nxt, kv_cache=cache)
            nxt = logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
        return (time.time() - t0) / iters * 1000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "bf16"])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 10, 20])
    ap.add_argument("--with-compile", action="store_true",
                    help="also run the full-model and KDA-only compile probes (slow)")
    ap.add_argument("--out", type=Path, default=Path("evals/bench/decode_step.jsonl"))
    args = ap.parse_args()

    model, step = load_variant_from_run(args.run, device="cuda")
    if args.dtype == "bf16":
        model = model.to(torch.bfloat16)
    model = model.eval()

    rec: dict = {"run": str(args.run), "step": step, "dtype": args.dtype,
                 "eager_ms_by_batch": {}}
    for B in args.batches:
        ms = step_ms(model, B)
        rec["eager_ms_by_batch"][str(B)] = round(ms, 1)
        print(f"eager B={B}: {ms:.1f} ms/step", flush=True)

    if args.with_compile:
        for blk in model.transformer.h:
            if getattr(blk, "is_linear", False):
                blk.compile(dynamic=False)
        ms = step_ms(model, 10)
        rec["kda_compiled_ms_b10"] = round(ms, 1)
        print(f"KDA-blocks compiled B=10: {ms:.1f} ms/step", flush=True)
        # Full-model compile: measured over few iters — each step builds a NEW graph
        # (the cache's sliced KV shapes grow), so per-step cost is dominated by
        # recompilation. That pathology is the result being recorded.
        compiled = torch.compile(model, dynamic=False)
        cache = build_cache(model, 10, "cuda", dtype=next(model.parameters()).dtype)
        idx = torch.randint(100, 5000, (10, 600), device="cuda")
        with torch.no_grad():
            logits, _ = model(idx, kv_cache=cache)
            nxt = logits[:, -1:].argmax(-1)
            logits, _ = compiled(nxt, kv_cache=cache)  # first compile, not timed
            nxt = logits[:, -1:].argmax(-1)
            torch.cuda.synchronize()
            t0 = time.time()
            iters = 5
            for _ in range(iters):
                logits, _ = compiled(nxt, kv_cache=cache)
                nxt = logits[:, -1:].argmax(-1)
            torch.cuda.synchronize()
            ms = (time.time() - t0) / iters * 1000
        rec["full_compiled_ms_b10"] = round(ms, 1)
        print(f"full-model compiled B=10: {ms:.1f} ms/step", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
