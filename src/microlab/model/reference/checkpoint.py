"""Load a trained VariantGPT from a run directory's latest checkpoint. Shared by the
interp report, the inference bench, and the console's serving endpoint."""

from __future__ import annotations

from pathlib import Path

import torch

from microlab.model.reference.variants import VariantConfig, VariantGPT


def load_variant_from_run(run_dir: Path, device: str = "cpu") -> tuple[VariantGPT, int]:
    """Latest ckpt_*.pt by step number. Raises FileNotFoundError when none exists."""
    run_dir = Path(run_dir)
    ckpts = sorted(run_dir.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        raise FileNotFoundError(f"no ckpt_*.pt in {run_dir}")
    ckpt = torch.load(ckpts[-1], map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = VariantGPT(VariantConfig(
        vocab_size=cfg.vocab_size, block_size=cfg.block_size, n_layer=cfg.n_layer,
        n_head=cfg.n_head, n_embd=cfg.n_embd, dropout=0.0, norm=cfg.norm, pos=cfg.pos,
        mlp=cfg.mlp,
    ))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt["step"]
