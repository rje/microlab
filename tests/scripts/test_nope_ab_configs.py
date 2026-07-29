"""configs/nope-ab-{rope,nope}.py: the NoPE-vs-RoPE A/B arms must be IDENTICAL in every
field except `pos` and `out_dir`, and must reuse the muon-ab sizing wholesale (same
~124M geometry, Muon optimizer, batch/accum/steps/eval cadence) so curves are comparable
across the two A/B experiments."""

import importlib.util
from dataclasses import fields
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _load(path):
    spec = importlib.util.spec_from_file_location("cfg_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.config


def test_arms_differ_only_in_pos_and_out_dir():
    rope = _load(_ROOT / "configs" / "nope-ab-rope.py")
    nope = _load(_ROOT / "configs" / "nope-ab-nope.py")
    diff = {f.name for f in fields(type(rope))
            if getattr(rope, f.name) != getattr(nope, f.name)}
    assert diff == {"pos", "out_dir"}
    assert rope.pos == "rope" and nope.pos == "nope"
    assert rope.out_dir == "runs/nope-ab-rope"
    assert nope.out_dir == "runs/nope-ab-nope"


def test_arms_reuse_muon_ab_sizing():
    arm = _load(_ROOT / "configs" / "nope-ab-rope.py")
    muon = _load(_ROOT / "configs" / "muon-ab-muon.py")
    same = ("vocab_size", "block_size", "n_layer", "n_head", "n_embd", "dropout", "norm",
            "mlp", "n_kv_head", "rope_base", "optimizer", "muon_lr", "lr", "min_lr",
            "weight_decay", "grad_clip", "warmup_steps", "max_steps", "lr_decay_steps",
            "batch_size", "grad_accum", "compile", "compile_mode", "eval_interval",
            "eval_iters", "ckpt_interval", "ckpt_keep", "log_interval", "dtype", "seed")
    for f in same:
        assert getattr(arm, f) == getattr(muon, f), f
    assert arm.block_size == 1024
    assert arm.optimizer == "muon"
    assert arm.pos == "rope"
