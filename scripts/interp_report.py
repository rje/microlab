"""Interp report against a trained checkpoint: logit-lens progression for a prompt,
per-head induction scores, and attention heatmaps for the top induction heads.

    python scripts/interp_report.py runs/150m --data-dir data/shards/tinystories \
        --prompt "Once upon a time" --out runs/interp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from microlab.interp.reference.lens import (  # noqa: E402
    attention_patterns,
    collect_residual_stream,
    induction_score,
    logit_lens,
    repeated_token_sequence,
)
from microlab.model.reference.variants import VariantGPT  # noqa: E402
from microlab.tokenizer.fast import FastTokenizer  # noqa: E402


def load_model(run_dir: Path) -> VariantGPT:
    ckpts = sorted(run_dir.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        raise FileNotFoundError(f"no ckpt_*.pt in {run_dir}")
    ckpt = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    from microlab.model.reference.variants import VariantConfig

    model = VariantGPT(VariantConfig(
        vocab_size=cfg.vocab_size, block_size=cfg.block_size, n_layer=cfg.n_layer,
        n_head=cfg.n_head, n_embd=cfg.n_embd, dropout=0.0, norm=cfg.norm, pos=cfg.pos,
        mlp=cfg.mlp,
    ))
    model.load_state_dict(ckpt["model"])
    print(f"loaded {ckpts[-1]} (step {ckpt['step']})")
    return model.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--data-dir", default="data/shards/tinystories")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--out", type=Path, default=Path("runs/interp"))
    args = ap.parse_args()

    tok = FastTokenizer.load(str(Path(args.data_dir) / "tokenizer.json"))
    model = load_model(args.run_dir)
    args.out.mkdir(parents=True, exist_ok=True)

    ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long)
    res = collect_residual_stream(model, ids)
    lens = logit_lens(res, model.transformer.ln_f, model.lm_head)
    print("\nlogit lens — top-1 next-token prediction per layer (last position):")
    for layer, row in enumerate(lens[:, 0, -1, :]):
        top = row.argmax().item()
        print(f"  layer {layer:>2}: {tok.decode([top])!r}  (p={row.softmax(-1).max():.3f})")

    g = torch.Generator().manual_seed(0)
    seq = repeated_token_sequence(model.config.vocab_size, period=32, repeats=2, generator=g)
    attn = attention_patterns(model, seq)
    scores = induction_score(attn, 32)  # (n_layer, n_head)
    flat = [(s.item(), layer, h) for layer, row in enumerate(scores) for h, s in enumerate(row)]
    flat.sort(reverse=True)
    print("\ntop induction heads (score, layer, head):")
    for s, layer, h in flat[:5]:
        print(f"  {s:.3f}  L{layer} H{h}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for rank, (s, layer, h) in enumerate(flat[:3]):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(attn[layer, h].numpy(), cmap="viridis")
            ax.set_title(f"L{layer} H{h} induction={s:.3f}")
            fig.savefig(args.out / f"induction_{rank}_L{layer}H{h}.png", dpi=120)
            plt.close(fig)
        print(f"\nheatmaps -> {args.out}/")
    except ImportError:
        print("\nmatplotlib not installed — skipped heatmaps (scores above still stand)")


if __name__ == "__main__":
    main()
