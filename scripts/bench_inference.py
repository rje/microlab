"""Inference bench against a trained checkpoint: tok/s uncached vs KV-cached vs
cached+int8, perplexity before/after quantization, and the GQA cache-size table.

    python scripts/bench_inference.py runs/150m --data-dir data/shards/tinystories
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from microlab.data.shard_dataset import ShardDataset  # noqa: E402
from microlab.evals.perplexity import evaluate_perplexity  # noqa: E402
from microlab.infer.reference.kv_cache import generate_cached  # noqa: E402
from microlab.infer.reference.quant import quantize_model_  # noqa: E402
from microlab.model.reference.sample import generate  # noqa: E402
from microlab.model.reference.variants import VariantConfig, VariantGPT  # noqa: E402


def load_model(run_dir: Path) -> VariantGPT:
    ckpts = sorted(run_dir.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        raise FileNotFoundError(f"no ckpt_*.pt in {run_dir}")
    ckpt = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = VariantGPT(VariantConfig(
        vocab_size=cfg.vocab_size, block_size=cfg.block_size, n_layer=cfg.n_layer,
        n_head=cfg.n_head, n_embd=cfg.n_embd, dropout=0.0, norm=cfg.norm, pos=cfg.pos,
        mlp=cfg.mlp,
    ))
    model.load_state_dict(ckpt["model"])
    print(f"loaded {ckpts[-1]} (step {ckpt['step']})")
    return model.eval()


def bench(fn, *args, n=3, **kwargs) -> float:
    # tok/s counts the prompt tokens too, equally on both sides of the comparison — the
    # cached/uncached ratio is exact; absolute figures are ~3% inflated (8 / 264 tokens).
    fn(*args, **kwargs)  # warmup
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn(*args, **kwargs)
    dt = (time.perf_counter() - t0) / n
    return (out.size(1) * out.size(0)) / dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--data-dir", default="data/shards/tinystories")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--new-tokens", type=int, default=256)
    args = ap.parse_args()

    model = load_model(args.run_dir).to(args.device)
    idx = torch.zeros((1, 8), dtype=torch.long, device=args.device)

    tps_slow = bench(generate, model, idx, args.new_tokens, temperature=0.0)
    tps_fast = bench(generate_cached, model, idx, args.new_tokens, temperature=0.0)
    print(f"uncached: {tps_slow:8.1f} tok/s")
    print(f"KV cache: {tps_fast:8.1f} tok/s  ({tps_fast / tps_slow:.1f}x)")

    val = ShardDataset(args.data_dir, split="val")
    ppl = evaluate_perplexity(model, val, model.config.block_size, 8, iters=50,
                              device=args.device)
    q8 = quantize_model_(copy.deepcopy(model), bits=8)
    q4 = quantize_model_(copy.deepcopy(model), bits=4)
    ppl8 = evaluate_perplexity(q8, val, model.config.block_size, 8, iters=50,
                               device=args.device)
    ppl4 = evaluate_perplexity(q4, val, model.config.block_size, 8, iters=50,
                               device=args.device)
    print(f"perplexity: fp32={ppl:.2f}  int8={ppl8:.2f}  int4={ppl4:.2f}")

    cfg = model.config
    hd = cfg.n_embd // cfg.n_head
    for n_kv in (cfg.n_head, max(1, cfg.n_head // 2), max(1, cfg.n_head // 4), 1):
        by = 2 * cfg.n_layer * n_kv * cfg.block_size * hd * 2  # k+v, bf16 bytes
        print(f"KV cache @ n_kv_head={n_kv:>2}: {by / 1e6:7.1f} MB per sequence")


if __name__ == "__main__":
    main()
