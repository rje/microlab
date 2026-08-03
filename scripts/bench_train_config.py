#!/usr/bin/env python
"""Sweep training-config levers for throughput and memory, on the REAL trainer.

    python scripts/bench_train_config.py --config configs/coder-1b.py

Why this exists: the 1B launched projecting ~25 days on the assumption that compile gives
~2x, a figure carried over from configs/1b.py — a DENSE MHA model at block_size 1024.
Measured on this architecture at 32k it gave nothing (4,835 tok/s compiled vs 4,772
uncompiled), and the real projection was 50 days. That is a number that should have been
measured before a multi-week commitment, not assumed across a regime change.

Each variant runs the actual pretrain loop (Muon, fused CE, gradient checkpointing as
configured) for a few steps and reports tokens/sec and peak memory read back from the run's
own telemetry. Nothing here is modelled.

tokens/sec is the comparison metric because it normalises across block_size and batch_size,
so a 16k row and a 32k row are directly comparable per token of training budget.
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCRATCH = Path("/tmp/claude-1000/-home-rje-src-python-microlab/"
               "7e96a61d-f6ef-46c5-b569-f265e8a1530b/scratchpad/bench")

# (block_size, batch_size, grad_checkpoint, compile)
VARIANTS = [
    (32768, 1, True,  False),   # what the run launched with (minus compile)
    (32768, 1, False, False),   # 19 GB of headroom said this was worth trying
    (32768, 2, True,  False),
    (32768, 2, False, False),
    (16384, 2, True,  False),
    (16384, 4, True,  False),
    (8192,  4, True,  False),
    (8192,  8, True,  False),
]


def read_telemetry(out_dir: Path) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    fs = sorted(glob.glob(str(out_dir / "events.out.tfevents.*")))
    if not fs:
        return {}
    ea = EventAccumulator(fs[-1], size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags()["scalars"]
    out = {}
    if "train/tokens_per_sec" in tags:
        vals = [s.value for s in ea.Scalars("train/tokens_per_sec")]
        out["tok_s"] = max(vals)           # steady state, not the first warm-up step
    if "gpu/mem_max_allocated_gb" in tags:
        out["peak_gb"] = max(s.value for s in ea.Scalars("gpu/mem_max_allocated_gb"))
    if "train/loss" in tags:
        s = ea.Scalars("train/loss")
        out["last_loss"] = s[-1].value
        out["steps"] = s[-1].step
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--target-tokens", type=float, default=21e9,
                    help="budget used to project days-to-finish")
    ap.add_argument("--out", default="docs/train-config-bench.json")
    ap.add_argument("--variants", default=None,
                    help="semicolon-separated block,batch,ckpt,compile — e.g. "
                         "'32768,1,1,0;32768,1,0,0'. Defaults to the local 48 GB sweep; "
                         "an 80 GB card wants the no-checkpointing rows that OOM here.")
    a = ap.parse_args()

    global VARIANTS
    if a.variants:
        VARIANTS = []
        for spec in a.variants.split(";"):
            if not spec.strip():
                continue
            b, bs, ck, cp = (int(x) for x in spec.split(","))
            VARIANTS.append((b, bs, bool(ck), bool(cp)))

    SCRATCH.mkdir(parents=True, exist_ok=True)
    base = Path(a.config).read_text()
    results = []
    print(f"{'block':>7} {'bs':>3} {'ckpt':>5} {'comp':>5} {'tok/s':>9} {'peak GB':>8} "
          f"{'days@21B':>9}")
    for block, bs, ckpt, comp in VARIANTS:
        tag = f"b{block}_bs{bs}_{'ck' if ckpt else 'nock'}_{'c' if comp else 'noc'}"
        out_dir = SCRATCH / tag
        shutil.rmtree(out_dir, ignore_errors=True)
        cfg = base
        for old, new in (
            (f"block_size={32768}", f"block_size={block}"),
            ("batch_size=1", f"batch_size={bs}"),
            ("grad_accum=16", f"grad_accum={a.grad_accum}"),
            ("grad_checkpoint=True", f"grad_checkpoint={ckpt}"),
            ("compile=True", f"compile={comp}"),
            ("max_steps=40000", f"max_steps={a.steps}"),
            ("lr_decay_steps=40000", f"lr_decay_steps={a.steps}"),
            ("warmup_steps=700", "warmup_steps=2"),
            ("eval_interval=500", "eval_interval=100000"),
            ("ckpt_interval=250", "ckpt_interval=100000"),
            ("ckpt_milestone_interval=2000", "ckpt_milestone_interval=100000"),
            ("log_interval=25", "log_interval=1"),
            ('out_dir="runs/coder-1b"', f'out_dir="{out_dir}"'),
        ):
            cfg = cfg.replace(old, new)
        p = SCRATCH / f"{tag}.py"
        p.write_text(cfg)

        t0 = time.time()
        proc = subprocess.run([sys.executable, "scripts/pretrain.py", str(p)],
                              capture_output=True, text=True, timeout=3600)
        tel = read_telemetry(out_dir)
        if proc.returncode != 0 or "tok_s" not in tel:
            err = (proc.stderr or proc.stdout)[-200:].replace("\n", " ")
            oom = "OutOfMemoryError" in proc.stderr or "out of memory" in proc.stderr
            note = "OOM" if oom else "FAILED"
            print(f"{block:>7,} {bs:>3} {str(ckpt):>5} {str(comp):>5} {note:>9} "
                  f"{'-':>8} {'-':>9}  {err[:60]}")
            results.append({"block": block, "batch_size": bs, "grad_checkpoint": ckpt,
                            "compile": comp, "status": note, "error": err})
            continue
        days = a.target_tokens / tel["tok_s"] / 86400
        print(f"{block:>7,} {bs:>3} {str(ckpt):>5} {str(comp):>5} {tel['tok_s']:>9,.0f} "
              f"{tel['peak_gb']:>8.1f} {days:>9.1f}   ({time.time()-t0:.0f}s)")
        results.append({"block": block, "batch_size": bs, "grad_checkpoint": ckpt,
                        "compile": comp, "status": "ok", **tel, "days_at_target": days})
        shutil.rmtree(out_dir, ignore_errors=True)

    Path(a.out).write_text(json.dumps(
        {"target_tokens": a.target_tokens, "grad_accum": a.grad_accum,
         "steps_per_variant": a.steps, "results": results}, indent=1))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
