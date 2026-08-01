"""configs/nope-audit-lr{05,10,20}.py: the NoPE-verdict-audit LR-fairness sweep. Each
arm must be IDENTICAL to configs/nope-ab-nope.py except muon_lr (x0.5 / x1 / x2),
max_steps=1000, and out_dir — same seed, same data order, same warmup+cosine SCHEDULE
(lr_decay_steps stays 4500) so the x1 arm replays the original's first 1000 steps
exactly and doubles as a run-to-run stability point."""

import importlib.util
from dataclasses import fields
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]

ARMS = {"nope-audit-lr05": 0.01, "nope-audit-lr10": 0.02, "nope-audit-lr20": 0.04}


def _load(path):
    spec = importlib.util.spec_from_file_location("cfg_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.config


@pytest.mark.parametrize("name,muon_lr", sorted(ARMS.items()))
def test_sweep_arm_differs_from_original_only_in_lr_steps_out_dir(name, muon_lr):
    base = _load(_ROOT / "configs" / "nope-ab-nope.py")
    arm = _load(_ROOT / "configs" / f"{name}.py")
    diff = {f.name for f in fields(type(base))
            if getattr(base, f.name) != getattr(arm, f.name)}
    allowed = {"max_steps", "out_dir"} | ({"muon_lr"} if muon_lr != base.muon_lr else set())
    assert diff == allowed, f"{name}: unexpected diffs {diff - allowed}"
    assert arm.muon_lr == muon_lr
    assert arm.max_steps == 1000
    assert arm.lr_decay_steps == 4500  # schedule SHAPE unchanged: truncated, not squeezed
    assert arm.seed == base.seed
    assert arm.pos == "nope"
    assert arm.out_dir == f"runs/{name}"


def test_sweep_covers_half_base_double():
    base = _load(_ROOT / "configs" / "nope-ab-nope.py")
    lrs = sorted(ARMS.values())
    assert lrs == [base.muon_lr * 0.5, base.muon_lr, base.muon_lr * 2.0]
