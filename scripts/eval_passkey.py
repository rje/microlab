"""Passkey retrieval eval: can the model recall a 5-digit key buried in filler text?

    python scripts/eval_passkey.py --run runs/1b --out evals/passkey/1b-base.json

The standard long-context probe (Mohtashami & Jaggi 2023, used by the PI and YaRN
papers): a prompt of EXACTLY --length tokens containing filler, a key sentence buried at
a controlled depth (0.1 = near the start, 0.9 = near the end), and a query at the end;
greedy-generate and score exact match of the digits. The grid (lengths x depths, --n
samples each with deterministic per-cell keys) maps where retrieval works and where it
dies — run it against the base model to show the pre-extension cliff, and against an
extended model to show the cliff moved.

Prompt lengths are controlled in TOKENS: component token streams (preamble / filler /
key sentence / query) are encoded separately and spliced, with the filler stream sliced
to make the total come out exact. Sliced filler can split a word mid-token-stream — the
same "windows cross boundaries" convention as base pretraining, and irrelevant to the
retrieval task.

Probing BEYOND the trained context: the model forward asserts T <= block_size, so the
checkpoint's model is rebuilt with its RoPE cache (and KV-cache capacity) extended to
cover the longest prompt. Weights are position-agnostic — nothing else changes — and
positions beyond the trained window behaving badly is exactly what this eval measures.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from microlab.infer.reference.kv_cache import generate_cached  # noqa: E402
from microlab.model.reference.checkpoint import latest_checkpoint  # noqa: E402
from microlab.model.reference.variants import VariantConfig, VariantGPT  # noqa: E402
from microlab.tokenizer.fast import FastTokenizer  # noqa: E402

PREAMBLE = (
    "There is important information hidden inside a lot of irrelevant text. Find it and "
    "memorize it. I will quiz you about the important information there. "
)
FILLER = (
    "The grass is green. The sky is blue. The sun is yellow. Here we go. "
    "There and back again. "
)
KEY_SENTENCE = "The pass key is {key}. Remember it. {key} is the pass key. "
QUERY = "\nWhat is the pass key? The pass key is"


def build_passkey_prompt(tok, length: int, depth: float, key: str) -> list[int]:
    """Token ids of EXACTLY `length` tokens: preamble + filler with the key sentence
    spliced in at `depth` (fraction of the filler that precedes it) + the query tail."""
    if not 0.0 <= depth <= 1.0:
        raise ValueError(f"depth must be in [0, 1] (got {depth})")
    if not key.isdigit():
        raise ValueError(f"key must be digits (got {key!r})")
    pre_ids = tok.encode(PREAMBLE)
    key_ids = tok.encode(KEY_SENTENCE.format(key=key))
    query_ids = tok.encode(QUERY)
    n_filler = length - len(pre_ids) - len(key_ids) - len(query_ids)
    if n_filler < 0:
        raise ValueError(
            f"length {length} too short: preamble+key+query alone take "
            f"{len(pre_ids) + len(key_ids) + len(query_ids)} tokens")
    filler_unit = tok.encode(FILLER)
    reps = n_filler // len(filler_unit) + 1
    filler_stream = filler_unit * reps
    n_before = round(depth * n_filler)
    ids = (pre_ids + filler_stream[:n_before] + key_ids
           + filler_stream[:n_filler - n_before] + query_ids)
    assert len(ids) == length, f"prompt assembly bug: {len(ids)} != {length}"
    return ids


def extract_key(text: str) -> str | None:
    """First run of digits in the generated text, or None if it produced no digits."""
    m = re.search(r"\d+", text)
    return m.group(0) if m else None


def draw_key(seed: int, length: int, depth: float, sample: int) -> str:
    """Deterministic 5-digit key for one (cell, sample): same arguments -> same key."""
    rng = random.Random(f"{seed}:{length}:{depth}:{sample}")
    return str(rng.randint(10000, 99999))


def run_cell(model, tok, device: str, length: int, depth: float, n: int, seed: int,
             max_new: int) -> dict:
    """Greedy-generate and exact-match score `n` samples for one (length, depth) cell."""
    samples = []
    for i in range(n):
        key = draw_key(seed, length, depth, i)
        ids = build_passkey_prompt(tok, length, depth, key)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = generate_cached(model, idx, max_new, temperature=0.0)
        text = tok.decode(out[0, length:].tolist())
        answer = extract_key(text)
        samples.append({"key": key, "answer": answer, "text": text, "ok": answer == key})
    correct = sum(s["ok"] for s in samples)
    return {"length": length, "depth": depth, "n": n, "correct": correct,
            # n=0 is the documented way to skip the passkey grid and keep only the
            # loss-vs-length curve; acc is undefined rather than 0.0 in that case
            # (reporting 0.0 would read as "the model failed every probe").
            "acc": (correct / n) if n else None, "samples": samples}


def format_table(cells: list[dict]) -> str:
    """Accuracy grid, lengths down, depths across."""
    lengths = sorted({c["length"] for c in cells})
    depths = sorted({c["depth"] for c in cells})
    acc = {(c["length"], c["depth"]): c["acc"] for c in cells}
    header = "length  " + "".join(f"depth={d:<8g}" for d in depths)
    lines = [header]
    for length in lengths:
        row = f"{length:<8d}"
        for d in depths:
            a = acc.get((length, d))
            row += f"{a:<14.2f}" if a is not None else f"{'-':<14}"
        lines.append(row.rstrip())
    return "\n".join(lines)


def load_for_eval(run_dir: Path, min_context: int, device: str):
    """The run's latest checkpoint, rebuilt with the RoPE cache extended to cover
    `min_context` positions when that exceeds the trained block_size. Weights are
    position-agnostic (embeddings/attention/MLP shapes don't depend on context length),
    so the strict state-dict load is unchanged; only the non-persistent cos/sin buffers
    and the assert-guard/KV-cache capacity grow. rope_base stays the checkpoint's own
    (native theta, no ABF). pos="nope" checkpoints have no positional state at all —
    the rebuild just raises block_size (nothing to extend).

    Returns (model, step, ckpt_cfg, eval_block)."""
    ckpt_path = latest_checkpoint(run_dir)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    eval_block = max(cfg.block_size, min_context)
    model = VariantGPT(VariantConfig(
        vocab_size=cfg.vocab_size, block_size=eval_block, n_layer=cfg.n_layer,
        n_head=cfg.n_head, n_embd=cfg.n_embd, dropout=0.0, norm=cfg.norm, pos=cfg.pos,
        mlp=cfg.mlp, n_kv_head=getattr(cfg, "n_kv_head", None),
        rope_base=getattr(cfg, "rope_base", 10000.0),
        # Every architecture field a checkpoint may carry has to be threaded through here
        # or the strict state-dict load fails. This loader had drifted: without block_norm
        # it could load NO Peri-LN run, and without hybrid_every no GDN hybrid. Same
        # failure class as docs/sota-parity-1b.md #8 (a reference capability unreachable
        # from a caller). Keep in sync with
        # microlab.model.reference.checkpoint.load_variant_from_run.
        block_norm=getattr(cfg, "block_norm", "pre"),
        hybrid_every=getattr(cfg, "hybrid_every", None),
        gdn_chunk=getattr(cfg, "gdn_chunk", 64),
        gdn_conv_kernel=getattr(cfg, "gdn_conv_kernel", 4),
    ))
    model.load_state_dict(ckpt["model"])
    extended = (" (NoPE: no positional state to extend; only the length guard and "
                "KV-cache capacity grow)" if cfg.pos == "nope"
                else " (RoPE cache extended for the probe; weights unchanged)")
    print(f"loaded {ckpt_path.name} (step {ckpt['step']}): trained block_size "
          f"{cfg.block_size}, pos {cfg.pos}, rope_base "
          f"{getattr(cfg, 'rope_base', 10000.0):g}, eval context {eval_block}"
          + (extended if eval_block > cfg.block_size else ""), flush=True)
    return model.to(device).eval(), ckpt["step"], cfg, eval_block


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run dir with ckpt_*.pt + tokenizer.json")
    ap.add_argument("--out", required=True, type=Path,
                    help="JSON report path (do NOT point into an existing run dir)")
    ap.add_argument("--lengths", default="512,1024,2048,3072,4000",
                    help="comma-separated prompt lengths in tokens")
    ap.add_argument("--depths", default="0.1,0.5,0.9",
                    help="comma-separated key depths in [0,1]")
    ap.add_argument("--n", type=int, default=10, help="samples per (length, depth) cell")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=12)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device {args.device!r} requested but CUDA is unavailable")
    lengths = [int(x) for x in args.lengths.split(",")]
    depths = [float(x) for x in args.depths.split(",")]

    run_dir = Path(args.run)
    torch.set_float32_matmul_precision("high")  # TF32 for the fp32 prefill matmuls
    model, step, cfg, eval_block = load_for_eval(
        run_dir, max(lengths) + args.max_new + 16, args.device)
    tok = FastTokenizer.load(str(run_dir / "tokenizer.json"))

    t0 = time.time()
    cells: list[dict] = []
    report = {
        "run": str(run_dir), "ckpt": latest_checkpoint(run_dir).name, "step": step,
        "trained_block_size": cfg.block_size,
        "rope_base": getattr(cfg, "rope_base", 10000.0),
        "eval_block": eval_block, "lengths": lengths, "depths": depths,
        "n": args.n, "seed": args.seed, "max_new": args.max_new, "cells": cells,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for length in lengths:
        for depth in depths:
            cell = run_cell(model, tok, args.device, length, depth, args.n, args.seed,
                            args.max_new)
            cells.append(cell)
            # progressive write: the report on disk is always current up to the last cell
            args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"L={length} depth={depth}: {cell['correct']}/{cell['n']} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    print("\n" + format_table(cells))
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
