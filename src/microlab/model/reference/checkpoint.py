"""Load a trained VariantGPT from a run directory's latest checkpoint. Shared by the
interp report, the inference bench, and the console's serving endpoint."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

from microlab.model.reference.variants import VariantConfig, VariantGPT


def variant_config_from_ckpt(cfg, **overrides) -> VariantConfig:
    """Rebuild a VariantConfig from a checkpoint's stored config.

    Fields are enumerated from the dataclass rather than listed by hand. The hand-written
    list this replaces drifted TWICE: first missing `block_norm`/`hybrid_every` (so it
    could load no Peri-LN or hybrid run), then missing all five frontier fields —
    `gdn_gate`, `global_attn`, `mla_kv_lora`, `qk_norm`, `gdn_fused` — which made every
    eval script unable to load a frontier checkpoint at all. The failure was a shape
    mismatch deep in a strict state-dict load, far from the cause.

    A field the checkpoint lacks falls back to the dataclass default, which is correct and
    deliberate: checkpoints predating a field must rebuild as that era's model. What makes
    this safe rather than a silent guess is `test_variant_config_round_trips_every_field`,
    which fails the moment a new field can be lost in transit.
    """
    out = {f.name: getattr(cfg, f.name)
           for f in dataclasses.fields(VariantConfig) if hasattr(cfg, f.name)}
    out["dropout"] = 0.0                       # inference: never inherit train-time dropout
    out.update(overrides)
    return VariantConfig(**out)


def latest_checkpoint(run_dir: Path) -> Path:
    """Newest ckpt_*.pt in run_dir by step number. Raises FileNotFoundError when none
    exists. Exposed so callers can report WHICH checkpoint file was picked."""
    run_dir = Path(run_dir)
    ckpts = sorted(run_dir.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        raise FileNotFoundError(f"no ckpt_*.pt in {run_dir}")
    return ckpts[-1]


def resolve_checkpoint(run_dir: Path, step: int | None = None) -> Path:
    """`ckpt_<step>.pt` when `step` is given, else the newest.

    Evaluating a TRAJECTORY needs this: the interesting question during a multi-week
    pretrain is when a capability appears, and answering it means pointing an eval at a
    specific milestone rather than at whatever is newest. Missing steps raise and list
    what IS available, since a silent fall back to "latest" would silently re-evaluate
    the same checkpoint under a different label."""
    if step is None:
        return latest_checkpoint(run_dir)
    p = Path(run_dir) / f"ckpt_{step}.pt"
    if not p.exists():
        have = sorted(int(q.stem.split("_")[1])
                      for q in Path(run_dir).glob("ckpt_*.pt"))
        raise FileNotFoundError(
            f"no ckpt_{step}.pt in {run_dir}; available steps: {have}")
    return p


def load_variant_from_run(run_dir: Path, device: str = "cpu",
                          step: int | None = None) -> tuple[VariantGPT, int]:
    """Latest ckpt_*.pt by step number. Raises FileNotFoundError when none exists.

    Loads to CPU and moves only the model to ``device``. The checkpoint bundles the optimizer
    state (Adam m/v, ~2x the model size); mapping the whole file straight onto CUDA would spike
    that onto the GPU too (~11GB for the 1B), which can OOM a training run sharing the device.
    Inference never needs the optimizer state, so it stays on CPU and is freed with ``ckpt``."""
    ckpt = torch.load(resolve_checkpoint(run_dir, step), map_location="cpu",
                      weights_only=False)
    model = VariantGPT(variant_config_from_ckpt(ckpt["cfg"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt["step"]
