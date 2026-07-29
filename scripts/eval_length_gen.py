"""Length-generalization eval: val loss vs sequence length + passkey grid, per run dir.

    python scripts/eval_length_gen.py --run runs/nope-ab-nope \\
        --out evals/length_gen/nope-ab-nope.json

The measurement behind the NoPE-vs-RoPE ablation (Kazemnejad et al. 2305.19466; Kimi K3
ships globally-NoPE): does a model trained at block_size 1024 keep working when asked to
predict at 2048/4096?

(a) Teacher-forced val loss/ppl at each --lengths L on the fineweb val shards: windows of
    EXACTLY L tokens, per-position loss averaged over the FULL window (plus 512-token
    bucket means, which localize WHERE beyond-train-length loss explodes). Batches are
    drawn from a generator seeded per (seed, length), so both A/B arms score the same
    windows.

(b) The passkey retrieval grid (scripts/eval_passkey.py machinery) out to 4096.

The model is rebuilt ONCE with block_size = the largest context any probe needs
(eval_passkey.load_for_eval). For pos="rope" that extends the cos/sin cache at the
checkpoint's NATIVE theta — deliberately NO ABF/PI/YaRN: raw extrapolation is the honest
comparison. For pos="nope" there is no positional state at all; only the T <= block_size
guard (and KV-cache capacity) grows.

Output is one JSON per run, written progressively (the file on disk is always current up
to the last completed section). Compare two runs with scripts/analyze_nope_ab.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from microlab.data.shard_dataset import ShardDataset  # noqa: E402
from microlab.model.reference.checkpoint import latest_checkpoint  # noqa: E402
from microlab.tokenizer.fast import FastTokenizer  # noqa: E402

# scripts/ is not a package: load the passkey module (model rebuild + grid machinery)
# from the sibling file, same trick the tests use.
_SPEC = importlib.util.spec_from_file_location(
    "eval_passkey", Path(__file__).resolve().parent / "eval_passkey.py")
ep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ep)


# ------------------------------------------------------------------------ batch planning

def seqs_for_budget(budget_tokens: int, length: int) -> int:
    """Windows needed to cover ~budget_tokens at this length (ceil, never zero)."""
    return max(1, -(-budget_tokens // length))


def micro_batch_size(length: int, max_micro_tokens: int) -> int:
    """Sequences per forward such that one forward stays under max_micro_tokens."""
    return max(1, max_micro_tokens // length)


def batch_plan(n_seqs: int, micro_bs: int) -> list[int]:
    """Micro-batch sizes covering exactly n_seqs sequences."""
    if n_seqs <= 0 or micro_bs <= 0:
        raise ValueError(f"need positive n_seqs/micro_bs (got {n_seqs}, {micro_bs})")
    plan = [micro_bs] * (n_seqs // micro_bs)
    if n_seqs % micro_bs:
        plan.append(n_seqs % micro_bs)
    return plan


# --------------------------------------------------------------- per-position NLL logic

def position_nll_sums(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-position NLL summed over the batch: (B, T, V), (B, T) -> (T,) float64."""
    B, T, V = logits.shape
    nll = F.cross_entropy(logits.reshape(-1, V).float(), targets.reshape(-1),
                          reduction="none")
    return nll.reshape(B, T).sum(0).double()


def summarize_positions(sums: torch.Tensor, n_seqs: int, bucket: int) -> dict:
    """Full-window mean loss/ppl + bucket means from per-position NLL sums.

    Every window is full (all positions seen n_seqs times), so the full-window mean is
    the plain mean of per-position means. Buckets must tile the window exactly."""
    T = sums.numel()
    if T % bucket != 0:
        raise ValueError(f"length {T} is not a multiple of bucket {bucket}")
    means = sums / n_seqs
    mean_loss = means.mean().item()
    bucket_means = means.reshape(T // bucket, bucket).mean(dim=1).tolist()
    return {"mean_loss": mean_loss, "ppl": math.exp(mean_loss),
            "bucket_size": bucket, "bucket_means": bucket_means}


@torch.no_grad()
def eval_loss_at_length(model, dataset, length: int, n_seqs: int, micro_bs: int,
                        device: str, seed: int, bucket: int = 512) -> dict:
    """Teacher-forced loss over n_seqs windows of exactly `length` tokens.

    The generator is seeded per (seed, length): both A/B arms draw identical windows,
    and per-length draws don't interact. bf16 autocast on CUDA matches the training
    trainer's eval numerics; the NLL itself is computed in fp32/64."""
    gen = torch.Generator().manual_seed(seed * 1_000_003 + length)
    amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if str(device).startswith("cuda") else nullcontext())
    sums = torch.zeros(length, dtype=torch.float64)
    for bs in batch_plan(n_seqs, micro_bs):
        x, y = dataset.get_batch(length, bs, device, gen)
        with amp:
            logits, _ = model(x)
        sums += position_nll_sums(logits, y).cpu()
    out = {"length": length, "n_seqs": n_seqs, "tokens": n_seqs * length}
    out.update(summarize_positions(sums, n_seqs, bucket))
    return out


def format_loss_table(results: list[dict]) -> str:
    lines = [f"{'length':<8}{'mean_loss':<12}{'ppl':<12}bucket_means (per 512 positions)"]
    for r in results:
        buckets = " ".join(f"{b:.3f}" for b in r["bucket_means"])
        lines.append(f"{r['length']:<8d}{r['mean_loss']:<12.4f}{r['ppl']:<12.2f}{buckets}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run dir with ckpt_*.pt + tokenizer.json")
    ap.add_argument("--out", required=True, type=Path,
                    help="JSON report path (do NOT point into an existing run dir)")
    ap.add_argument("--data-dir", default="data/shards/fineweb-100bt",
                    help="shard dir with val split (same for both arms)")
    ap.add_argument("--lengths", default="512,1024,2048,4096",
                    help="comma-separated eval sequence lengths in tokens")
    ap.add_argument("--tokens-per-length", type=int, default=2_097_152,
                    help="~val tokens scored at each length (matches the 2M training eval)")
    ap.add_argument("--micro-tokens", type=int, default=65_536,
                    help="max tokens per forward (bounds logits memory)")
    ap.add_argument("--bucket", type=int, default=512,
                    help="per-position bucket size for localizing the blow-up")
    ap.add_argument("--passkey-lengths", default="512,1024,2048,3072,4096",
                    help="passkey grid prompt lengths ('' skips the passkey section)")
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--n", type=int, default=10, help="passkey samples per cell")
    ap.add_argument("--max-new", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device {args.device!r} requested but CUDA is unavailable")

    lengths = [int(x) for x in args.lengths.split(",")]
    pk_lengths = [int(x) for x in args.passkey_lengths.split(",")] \
        if args.passkey_lengths else []
    depths = [float(x) for x in args.depths.split(",")]
    run_dir = Path(args.run)
    torch.set_float32_matmul_precision("high")

    # one model build covers every probe (largest loss window or passkey prompt + headroom)
    min_context = max(lengths + [(pl + args.max_new + 16) for pl in pk_lengths])
    model, step, cfg, eval_block = ep.load_for_eval(run_dir, min_context, args.device)

    report: dict = {
        "run": str(run_dir), "ckpt": latest_checkpoint(run_dir).name, "step": step,
        "pos": cfg.pos, "trained_block_size": cfg.block_size,
        "rope_base": getattr(cfg, "rope_base", 10000.0), "eval_block": eval_block,
        "seed": args.seed,
        "loss": {"data_dir": args.data_dir, "tokens_per_length": args.tokens_per_length,
                 "lengths": lengths, "results": []},
        "passkey": {"lengths": pk_lengths, "depths": depths, "n": args.n,
                    "max_new": args.max_new, "cells": []},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def flush() -> None:  # progressive write: on-disk report is current per section
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    t0 = time.time()
    val_ds = ShardDataset(args.data_dir, split="val")
    for length in lengths:
        r = eval_loss_at_length(
            model, val_ds, length,
            n_seqs=seqs_for_budget(args.tokens_per_length, length),
            micro_bs=micro_batch_size(length, args.micro_tokens),
            device=args.device, seed=args.seed, bucket=args.bucket)
        report["loss"]["results"].append(r)
        flush()
        print(f"loss L={length}: {r['mean_loss']:.4f} (ppl {r['ppl']:.2f}) "
              f"[{time.time() - t0:.0f}s]", flush=True)
    print("\n" + format_loss_table(report["loss"]["results"]) + "\n", flush=True)

    if pk_lengths:
        tok = FastTokenizer.load(str(run_dir / "tokenizer.json"))
        for length in pk_lengths:
            for depth in depths:
                cell = ep.run_cell(model, tok, args.device, length, depth, args.n,
                                   args.seed, args.max_new)
                report["passkey"]["cells"].append(cell)
                flush()
                print(f"passkey L={length} depth={depth}: "
                      f"{cell['correct']}/{cell['n']} [{time.time() - t0:.0f}s]",
                      flush=True)
        print("\n" + ep.format_table(report["passkey"]["cells"]))
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
