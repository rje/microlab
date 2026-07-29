"""configs/periln-ab-{pre,peri}.py: the Peri-LN-vs-Pre-LN A/B arms must be IDENTICAL in
every field except `block_norm` and `out_dir`, and must reuse the muon-ab sizing
wholesale (same ~124M geometry, Muon optimizer, batch/accum/steps/eval cadence) so
curves are comparable across the lab's A/B experiments."""

import importlib.util
from dataclasses import fields
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _load(path):
    spec = importlib.util.spec_from_file_location("cfg_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.config


def test_arms_differ_only_in_block_norm_and_out_dir():
    pre = _load(_ROOT / "configs" / "periln-ab-pre.py")
    peri = _load(_ROOT / "configs" / "periln-ab-peri.py")
    diff = {f.name for f in fields(type(pre))
            if getattr(pre, f.name) != getattr(peri, f.name)}
    assert diff == {"block_norm", "out_dir"}
    assert pre.block_norm == "pre" and peri.block_norm == "peri"
    assert pre.out_dir == "runs/periln-ab-pre"
    assert peri.out_dir == "runs/periln-ab-peri"


def test_arms_reuse_muon_ab_sizing():
    arm = _load(_ROOT / "configs" / "periln-ab-pre.py")
    muon = _load(_ROOT / "configs" / "muon-ab-muon.py")
    same = ("vocab_size", "block_size", "n_layer", "n_head", "n_embd", "dropout", "norm",
            "pos", "mlp", "n_kv_head", "rope_base", "optimizer", "muon_lr", "lr",
            "min_lr", "weight_decay", "grad_clip", "warmup_steps", "max_steps",
            "lr_decay_steps", "batch_size", "grad_accum", "compile", "compile_mode",
            "eval_interval", "eval_iters", "ckpt_interval", "ckpt_keep", "log_interval",
            "dtype", "seed")
    for f in same:
        assert getattr(arm, f) == getattr(muon, f), f
    assert arm.block_size == 1024
    assert arm.optimizer == "muon"
    assert arm.block_norm == "pre"
