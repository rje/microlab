"""Phase-5 run-for-real (the stretch): score induction heads across EVERY saved
checkpoint of a run and plot the maximum induction score vs training step. Induction
heads — the attention heads that implement in-context copying, and with it most of
in-context learning — famously appear in a sharp phase change rather than gradually.
This driver reproduces that on a model whose every checkpoint you own.

    python scripts/induction_phase_change.py runs/150m --out runs/interp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from microlab.interp.reference.lens import (  # noqa: E402
    attention_patterns,
    induction_score,
    repeated_token_sequence,
)
from microlab.model.reference.checkpoint import load_variant_from_run  # noqa: E402
from microlab.model.reference.variants import VariantConfig, VariantGPT  # noqa: E402


def _load_ckpt(path: Path, device: str) -> VariantGPT:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = VariantGPT(VariantConfig(
        vocab_size=cfg.vocab_size, block_size=cfg.block_size, n_layer=cfg.n_layer,
        n_head=cfg.n_head, n_embd=cfg.n_embd, dropout=0.0, norm=cfg.norm, pos=cfg.pos,
        mlp=cfg.mlp,
    ))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("runs/interp"))
    ap.add_argument("--period", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpts = sorted(args.run_dir.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        raise FileNotFoundError(f"no ckpt_*.pt in {args.run_dir}")
    args.out.mkdir(parents=True, exist_ok=True)

    # A fixed repeated-token probe reused for every checkpoint so scores are comparable.
    final_model, final_step = load_variant_from_run(args.run_dir, device=args.device)
    vocab = final_model.config.vocab_size
    seq = repeated_token_sequence(vocab, args.period, args.repeats,
                                  torch.Generator().manual_seed(0)).to(args.device)

    rows = []
    for path in ckpts:
        step = int(path.stem.split("_")[1])
        model = _load_ckpt(path, args.device)
        attn = attention_patterns(model, seq)          # (n_layer, n_head, T, T)
        scores = induction_score(attn, args.period)    # (n_layer, n_head)
        best = scores.max().item()
        layer, head = [x.item() for x in torch.where(scores == scores.max())]
        rows.append((step, best, layer, head))
        print(f"step {step:>4}: max induction {best:.3f}  (L{layer} H{head})")

    steps = [r[0] for r in rows]
    best = [r[1] for r in rows]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(steps, best, marker="o")
        ax.set_xlabel("training step")
        ax.set_ylabel("max induction score (any head)")
        ax.set_title("Induction-head formation across training")
        ax.grid(True, alpha=0.3)
        out = args.out / "induction_phase_change.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"\nplot -> {out}")
    except ImportError:
        print("\nmatplotlib not installed — scores above stand, no plot written")


if __name__ == "__main__":
    main()
