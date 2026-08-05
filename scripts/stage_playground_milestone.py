#!/usr/bin/env python
"""Stage a milestone checkpoint as a Playground run: runs/<run>-step<N>/.

    python scripts/stage_playground_milestone.py --run coder-1b --step 4000

Pulls the checkpoint from B2 (the run prefix first, then the -trajectory archive), strips
it to WEIGHTS ONLY, and drops it where the console's run discovery already looks. Each
milestone becomes its own Playground entry, so the model can be prompted at successive
points in training — the qualitative face of the emergence trajectory the milestone
checkpoints exist to measure.

Weights-only on purpose: the full checkpoint is half optimizer state (Muon momentum,
AdamW moments) that generation never touches. Stripping halves disk and load time, and
the Playground copy can never be mistaken for a resumable training checkpoint — loading
it into the trainer fails loudly on the missing optimizer key.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def b2_client():
    import boto3
    from botocore.config import Config
    return boto3.client(
        "s3", endpoint_url=os.environ["B2_ENDPOINT"],
        aws_access_key_id=os.environ["B2_KEY_ID"],
        aws_secret_access_key=os.environ["B2_APPLICATION_KEY"],
        config=Config(retries={"max_attempts": 8, "mode": "adaptive"},
                      max_pool_connections=16))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="coder-1b")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--bucket", default="microlab-checkpoints")
    ap.add_argument("--root", default=".", help="repo root (console serves <root>/runs)")
    ap.add_argument("--tokenizer", default="data/shards/mix-v2/tokenizer.json")
    a = ap.parse_args()

    out = Path(a.root) / "runs" / f"{a.run}-step{a.step}"
    dest = out / f"ckpt_{a.step}.pt"
    if dest.exists():
        print(f"{dest} already staged")
        return 0
    out.mkdir(parents=True, exist_ok=True)

    s3 = b2_client()
    tmp = out / f".ckpt_{a.step}.download"
    got = None
    # The run prefix holds the recent window; the -trajectory archive holds what the
    # remote pruner retired. Try both — a milestone must be findable either way.
    for key in (f"{a.run}/ckpt_{a.step}.pt", f"{a.run}-trajectory/ckpt_{a.step}.pt"):
        try:
            t0 = time.time()
            s3.download_file(a.bucket, key, str(tmp))
            got = key
            print(f"pulled s3://{a.bucket}/{key} in {time.time() - t0:.0f}s")
            break
        except Exception:                            # noqa: BLE001 — try the next prefix
            continue
    if got is None:
        raise SystemExit(f"ckpt_{a.step}.pt not found under {a.run}/ or {a.run}-trajectory/")

    ck = torch.load(tmp, map_location="cpu", weights_only=False)
    slim = {"model": ck["model"], "cfg": ck["cfg"], "step": ck["step"],
            "playground_note": "weights-only staging; not resumable"}
    tmp2 = dest.with_suffix(".pt.tmp")
    torch.save(slim, tmp2)
    os.replace(tmp2, dest)
    tmp.unlink()
    full = sum(v.numel() for v in ck["model"].values() if hasattr(v, "numel"))
    print(f"staged {dest} ({dest.stat().st_size / 1e9:.1f} GB, {full/1e9:.2f}B params, "
          f"optimizer state stripped)")

    tok_dst = out / "tokenizer.json"
    if not tok_dst.exists():
        import shutil
        shutil.copy(a.tokenizer, tok_dst)
        print(f"tokenizer -> {tok_dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
