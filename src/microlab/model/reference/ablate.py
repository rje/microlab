"""Ablation runner: train the baseline and each single-flag variant on the same data
and seed, return a comparison of final loss / params / throughput."""

from __future__ import annotations

import torch

from microlab.model.reference.train import TrainConfig, train
from microlab.model.reference.variants import VariantConfig, VariantGPT

ABLATIONS: dict[str, dict] = {
    "baseline": {},
    "rmsnorm": {"norm": "rms"},
    "rope": {"pos": "rope"},
    "swiglu": {"mlp": "swiglu"},
}


def run_ablations(data: torch.Tensor, base: VariantConfig, train_cfg: TrainConfig,
                  ablations: dict[str, dict] | None = None,
                  val_data: torch.Tensor | None = None) -> dict[str, dict]:
    ablations = ablations if ablations is not None else ABLATIONS
    results: dict[str, dict] = {}
    for name, overrides in ablations.items():
        cfg = VariantConfig(**{**base.__dict__, **overrides})
        torch.manual_seed(train_cfg.seed)
        model = VariantGPT(cfg)
        stats = train(model, data, train_cfg, val_data=val_data)
        results[name] = {
            "final_loss": stats["final_loss"],
            "val_loss": stats["val_loss"],
            "params": model.num_params(),
            "tokens_per_sec": stats["tokens_per_sec"],
            "peak_vram_mb": stats["peak_vram_mb"],
        }
    return results
