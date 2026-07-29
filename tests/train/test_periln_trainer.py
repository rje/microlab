"""RunConfig -> Trainer -> VariantConfig plumbing for block_norm (Peri-LN lane). The
default must reproduce current behavior (no post-norm keys, old checkpoints unaffected);
block_norm="peri" must reach the model, train under the lab-standard Muon hybrid (the
new 1-D norm scales belong in the AdamW group), and round-trip through checkpoints,
including microlab.model.reference.checkpoint.load_variant_from_run."""

from __future__ import annotations

import pickle

import pytest
import torch

from microlab.model.reference.checkpoint import load_variant_from_run
from microlab.train.config import RunConfig
from microlab.train.trainer import TensorData, Trainer


def _cfg(**kw):
    base = dict(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32,
                warmup_steps=2, max_steps=3, lr_decay_steps=3, batch_size=4,
                eval_interval=1000, ckpt_interval=1000, device="cpu", dtype="float32")
    base.update(kw)
    return RunConfig(**base)


def _data():
    return TensorData(torch.randint(0, 64, (2000,), generator=torch.Generator().manual_seed(7)))


def test_default_runconfig_builds_pre_model():
    tr = Trainer(_cfg(), _data())
    for key in tr.raw_model.state_dict():
        assert "_post" not in key, f"default RunConfig grew state-dict key {key}"


def test_trainer_builds_peri_model():
    tr = Trainer(_cfg(block_norm="peri"), _data())
    block = tr.raw_model.transformer.h[0]
    assert hasattr(block, "ln_1_post") and hasattr(block, "ln_2_post")


def test_unknown_block_norm_raises_at_trainer():
    with pytest.raises(ValueError, match="block_norm"):
        Trainer(_cfg(block_norm="sandwich"), _data())


def test_peri_trains_with_muon_and_new_norms_stay_on_adamw():
    tr = Trainer(_cfg(block_norm="peri", optimizer="muon"), _data(), _data())
    post_ids = {id(p) for n, p in tr.raw_model.named_parameters() if "_post" in n}
    assert len(post_ids) == 2 * 2  # 2 layers x 2 post norms
    adamw_ids = {id(p) for g in tr.optimizer.adamw.param_groups for p in g["params"]}
    muon_ids = {id(p) for g in tr.optimizer.muon.param_groups for p in g["params"]}
    assert post_ids <= adamw_ids and not (post_ids & muon_ids)
    stats = tr.train()
    assert stats["step"] == 3 and stats["final_loss"] > 0


def test_peri_checkpoint_roundtrips_through_load_variant_from_run(tmp_path):
    cfg = _cfg(block_norm="peri", out_dir=str(tmp_path))
    tr = Trainer(cfg, _data())
    tr.train()
    path = tmp_path / "ckpt_3.pt"
    tr.save_checkpoint(str(path))
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    assert ckpt["cfg"].block_norm == "peri"
    # A fresh Trainer built from the saved cfg loads it (peri-shaped state dict).
    tr2 = Trainer(ckpt["cfg"], _data())
    tr2.load_checkpoint(str(path))
    for a, b in zip(tr.raw_model.parameters(), tr2.raw_model.parameters(), strict=True):
        assert torch.equal(a, b)
    # And the shared run-dir loader (interp/bench/serving path) rebuilds a peri model.
    model, step = load_variant_from_run(tmp_path)
    assert step == 3
    assert hasattr(model.transformer.h[0], "ln_1_post")
    got = dict(model.named_parameters())
    for name, p in tr.raw_model.named_parameters():
        assert torch.equal(p, got[name]), name


def test_old_pickled_cfg_gains_block_norm_default():
    # Checkpoints saved before block_norm existed unpickle without it in __dict__;
    # attribute access must fall back to the dataclass default ("pre").
    cfg = _cfg()
    cfg.__dict__.pop("block_norm")
    old = pickle.loads(pickle.dumps(cfg))
    assert old.block_norm == "pre"
