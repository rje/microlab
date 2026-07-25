"""pretrain.py --init-ckpt: warm-start a run from MODEL WEIGHTS ONLY (converted/foreign
checkpoint) — fresh optimizer, step 0, no RNG restore — mutually exclusive with an
out_dir that already has checkpoints. Authority rule: the run config wins; the strict
state-dict load asserts the checkpoint's weights fit the run-config model."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from microlab.train.config import RunConfig
from microlab.train.trainer import TensorData, Trainer

_SPEC = importlib.util.spec_from_file_location(
    "pretrain_script", Path(__file__).resolve().parents[2] / "scripts" / "pretrain.py")
pretrain = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pretrain)


def _cfg(**kw):
    base = dict(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32,
                warmup_steps=1, max_steps=2, lr_decay_steps=2, batch_size=4,
                eval_interval=1000, ckpt_interval=1000, device="cpu", dtype="float32")
    base.update(kw)
    return RunConfig(**base)


def _data(seed=11):
    return TensorData(torch.randint(0, 64, (2000,), generator=torch.Generator().manual_seed(seed)))


def _weights_ckpt(tmp_path, cfg, name="weights.pt", train_steps=2):
    """A donor run whose weights become the warm-start checkpoint (converted-ckpt format:
    model + cfg, no optimizer)."""
    donor = Trainer(cfg, _data())
    for _ in range(train_steps):
        donor.train_step()
    path = tmp_path / name
    torch.save({"model": donor.raw_model.state_dict(), "cfg": cfg}, path)
    return donor, path


def test_warm_start_loads_weights_fresh_optimizer_step0(tmp_path):
    donor, path = _weights_ckpt(tmp_path, _cfg(out_dir=str(tmp_path / "donor")))
    up = Trainer(_cfg(seed=99, out_dir=str(tmp_path / "up")), _data())
    pretrain.warm_start(up, str(path))
    for a, b in zip(up.raw_model.parameters(), donor.raw_model.parameters(), strict=True):
        assert torch.equal(a, b)
    assert up.step == 0
    assert up.optimizer.state_dict()["state"] == {}  # fresh AdamW: no moments carried over
    stats = up.train()  # and it actually trains from the warm start
    assert stats["step"] == 2


def test_warm_start_raises_when_out_dir_has_checkpoints(tmp_path):
    _, path = _weights_ckpt(tmp_path, _cfg(out_dir=str(tmp_path / "donor")))
    out_dir = tmp_path / "up"
    out_dir.mkdir()
    (out_dir / "ckpt_5.pt").touch()
    up = Trainer(_cfg(out_dir=str(out_dir)), _data())
    with pytest.raises(RuntimeError, match="already has checkpoints"):
        pretrain.warm_start(up, str(path))


def test_warm_start_run_cfg_wins_shape_mismatch_raises(tmp_path):
    # The checkpoint's cfg says n_embd=32; the run config says 48. Run cfg wins — the
    # model is built from it — and the strict load must fail loudly, not adapt.
    _, path = _weights_ckpt(tmp_path, _cfg(out_dir=str(tmp_path / "donor")))
    up = Trainer(_cfg(n_embd=48, n_head=3, out_dir=str(tmp_path / "up")), _data())
    with pytest.raises(RuntimeError, match="size mismatch"):
        pretrain.warm_start(up, str(path))


def test_warm_start_muon_optimizer_is_fresh(tmp_path):
    # The uptrain uses Muon; warm-starting must leave both sub-optimizers stateless.
    _, path = _weights_ckpt(tmp_path, _cfg(out_dir=str(tmp_path / "donor")))
    up = Trainer(_cfg(optimizer="muon", out_dir=str(tmp_path / "up")), _data())
    pretrain.warm_start(up, str(path))
    sd = up.optimizer.state_dict()
    assert sd["muon"]["state"] == {} and sd["adamw"]["state"] == {}
