"""Load a trained VariantGPT from a run directory's latest checkpoint. Shared by the
interp report, the inference bench, and the console's serving endpoint."""

from __future__ import annotations

from pathlib import Path

import torch

from microlab.model.reference.variants import VariantConfig, VariantGPT


def latest_checkpoint(run_dir: Path) -> Path:
    """Newest ckpt_*.pt in run_dir by step number. Raises FileNotFoundError when none
    exists. Exposed so callers can report WHICH checkpoint file was picked."""
    run_dir = Path(run_dir)
    ckpts = sorted(run_dir.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        raise FileNotFoundError(f"no ckpt_*.pt in {run_dir}")
    return ckpts[-1]


def load_variant_from_run(run_dir: Path, device: str = "cpu") -> tuple[VariantGPT, int]:
    """Latest ckpt_*.pt by step number. Raises FileNotFoundError when none exists.

    Loads to CPU and moves only the model to ``device``. The checkpoint bundles the optimizer
    state (Adam m/v, ~2x the model size); mapping the whole file straight onto CUDA would spike
    that onto the GPU too (~11GB for the 1B), which can OOM a training run sharing the device.
    Inference never needs the optimizer state, so it stays on CPU and is freed with ``ckpt``."""
    ckpt = torch.load(latest_checkpoint(run_dir), map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = VariantGPT(VariantConfig(
        vocab_size=cfg.vocab_size, block_size=cfg.block_size, n_layer=cfg.n_layer,
        n_head=cfg.n_head, n_embd=cfg.n_embd, dropout=0.0, norm=cfg.norm, pos=cfg.pos,
        mlp=cfg.mlp,
        # getattr: checkpoints from before these fields existed lack the attributes; the
        # defaults reproduce that era's model exactly (fused MHA, base 10000, pre-norm).
        n_kv_head=getattr(cfg, "n_kv_head", None),
        rope_base=getattr(cfg, "rope_base", 10000.0),
        block_norm=getattr(cfg, "block_norm", "pre"),
        hybrid_every=getattr(cfg, "hybrid_every", None),
        gdn_chunk=getattr(cfg, "gdn_chunk", 64),
        gdn_conv_kernel=getattr(cfg, "gdn_conv_kernel", 4),
    ))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt["step"]
