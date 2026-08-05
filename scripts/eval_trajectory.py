#!/usr/bin/env python
"""Sweep the FIXED prompt set across milestone checkpoints: the qualitative trajectory.

    python scripts/eval_trajectory.py --run coder-1b --steps 500,1000,1500,2000,4000

For each step: pull the checkpoint (run prefix, then -trajectory archive, then a staged
Playground copy — whichever exists), generate greedy completions for every prompt in
evals/trajectory_prompts.py, and append one JSONL row per (step, prompt) to
evals/trajectory/<run>-completions.jsonl. A markdown side-by-side lands next to it.

Greedy + fixed budgets so a difference between columns is a difference in the MODEL.
Rows are append-only and keyed (step, prompt id, prompt sha) — re-running a step skips
prompts already recorded, so the sweep is resumable and a later sweep extends the same
file. House rule: long jobs write progressively.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.trajectory_prompts import PROMPTS  # noqa: E402
from microlab.model.reference.checkpoint import variant_config_from_ckpt  # noqa: E402
from microlab.model.reference.variants import VariantGPT  # noqa: E402


def _psha(p: dict) -> str:
    return hashlib.sha256(p["text"].encode()).hexdigest()[:12]


def _fetch(run: str, step: int, cache_dir: Path) -> Path:
    """The checkpoint for `step`, from the cheapest source that has it."""
    local = cache_dir / f"ckpt_{step}.pt"
    if local.exists():
        return local
    staged = Path("runs") / f"{run}-step{step}" / f"ckpt_{step}.pt"
    if staged.exists():
        return staged
    import boto3
    from botocore.config import Config
    s3 = boto3.client(
        "s3", endpoint_url=os.environ["B2_ENDPOINT"],
        aws_access_key_id=os.environ["B2_KEY_ID"],
        aws_secret_access_key=os.environ["B2_APPLICATION_KEY"],
        config=Config(retries={"max_attempts": 8, "mode": "adaptive"},
                      max_pool_connections=16))
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = local.with_suffix(".part")
    for key in (f"{run}/ckpt_{step}.pt", f"{run}-trajectory/ckpt_{step}.pt"):
        try:
            t0 = time.time()
            s3.download_file("microlab-checkpoints", key, str(tmp))
            os.replace(tmp, local)
            print(f"  pulled {key} in {time.time() - t0:.0f}s", flush=True)
            return local
        except Exception:                            # noqa: BLE001 — try the next prefix
            continue
    raise SystemExit(f"ckpt_{step}.pt not found in B2 under {run}/ or {run}-trajectory/")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="coder-1b")
    ap.add_argument("--steps", required=True,
                    help="comma-separated milestone steps, e.g. 500,1000,2000")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tokenizer", default="data/shards/mix-v2/tokenizer.json")
    ap.add_argument("--out-dir", default="evals/trajectory")
    ap.add_argument("--cache-dir", default=None,
                    help="where pulled checkpoints land (default: <out-dir>/.ckpts)")
    a = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(a.cache_dir) if a.cache_dir else out_dir / ".ckpts"
    rows_path = out_dir / f"{a.run}-completions.jsonl"
    have = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            r = json.loads(line)
            have.add((r["step"], r["prompt_id"], r["prompt_sha"]))

    steps = sorted(int(s) for s in a.steps.split(","))
    for step in steps:
        todo = [p for p in PROMPTS if (step, p["id"], _psha(p)) not in have]
        if not todo:
            print(f"step {step}: all {len(PROMPTS)} prompts already recorded")
            continue
        ck_path = _fetch(a.run, step, cache_dir)
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model = VariantGPT(variant_config_from_ckpt(ck["cfg"]))
        model.load_state_dict(ck["model"])
        del ck
        model.to(a.device).eval()
        print(f"step {step}: {len(todo)} prompts", flush=True)
        with rows_path.open("a") as f:
            for p in todo:
                ids = torch.tensor([tok.encode(p["text"]).ids], device=a.device)
                t0 = time.time()
                with torch.no_grad():
                    from microlab.infer.reference.kv_cache import generate_cached
                    out = generate_cached(model, ids, p["n"], temperature=0.0)
                text = tok.decode(out[0].tolist())[len(p["text"]):]
                f.write(json.dumps({
                    "step": step, "prompt_id": p["id"], "prompt_sha": _psha(p),
                    "n_tokens": p["n"], "completion": text,
                    "gen_s": round(time.time() - t0, 1)}) + "\n")
                f.flush()
                print(f"  {p['id']}: {time.time() - t0:.0f}s", flush=True)
        del model
        if a.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # Side-by-side markdown, regenerated whole from the JSONL each sweep.
    rows = [json.loads(x) for x in rows_path.read_text().splitlines()]
    by_prompt: dict[str, dict[int, str]] = {}
    all_steps = sorted({r["step"] for r in rows})
    for r in rows:
        by_prompt.setdefault(r["prompt_id"], {})[r["step"]] = r["completion"]
    md = [f"# {a.run} — completion trajectory\n",
          "Greedy, fixed budgets; a difference between columns is the model changing.\n"]
    for p in PROMPTS:
        md.append(f"\n## `{p['id']}`\n\n```\n{p['text']}```\n")
        for s in all_steps:
            if s in by_prompt.get(p["id"], {}):
                md.append(f"\n<details><summary>step {s}</summary>\n\n```\n"
                          f"{by_prompt[p['id']][s]}\n```\n</details>\n")
    (out_dir / f"{a.run}-trajectory.md").write_text("".join(md))
    print(f"wrote {rows_path} and {out_dir / (a.run + '-trajectory.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
