"""Context-extension plumbing, end-to-end at tiny scale: a model built at an EXTENDED
block_size with a RAISED rope_base must warm-start from a shorter-context checkpoint via
pretrain.warm_start (weights are position-agnostic — no shape depends on context length),
run full-length forwards, stay causal beyond the old window, and generate past the old
block through the KV cache (whose capacity follows the new config). Also covers
eval_passkey.load_for_eval, which does the same extension for probing a frozen model."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from microlab.infer.reference.kv_cache import generate_cached
from microlab.model.reference.variants import build_rope_cache
from microlab.train.config import RunConfig
from microlab.train.trainer import TensorData, Trainer

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pretrain = _load_script("pretrain")
ep = _load_script("eval_passkey")

OLD_BLOCK, NEW_BLOCK = 16, 64
OLD_BASE, NEW_BASE = 10000.0, 100000.0


def _cfg(**kw):
    base = dict(vocab_size=64, block_size=OLD_BLOCK, n_layer=2, n_head=2, n_embd=32,
                rope_base=OLD_BASE, norm="rms", pos="rope", mlp="swiglu",
                warmup_steps=1, max_steps=2, lr_decay_steps=2, batch_size=4,
                eval_interval=1000, ckpt_interval=1000, device="cpu", dtype="float32")
    base.update(kw)
    return RunConfig(**base)


def _data(seed=11):
    return TensorData(torch.randint(0, 64, (2000,), generator=torch.Generator().manual_seed(seed)))


def _base_ckpt(tmp_path):
    """A short-context donor checkpoint (the tiny analog of runs/1b/ckpt_40000.pt)."""
    donor = Trainer(_cfg(out_dir=str(tmp_path / "donor")), _data())
    donor.train_step()
    path = tmp_path / "base.pt"
    torch.save({"model": donor.raw_model.state_dict(), "cfg": donor.cfg, "step": 1}, path)
    return donor, path


def _extended_trainer(tmp_path):
    return Trainer(_cfg(block_size=NEW_BLOCK, rope_base=NEW_BASE, seed=99,
                        out_dir=str(tmp_path / "ext")), _data())


def test_warm_start_short_ckpt_into_extended_model_and_full_length_forward(tmp_path):
    donor, path = _base_ckpt(tmp_path)
    ext = _extended_trainer(tmp_path)
    pretrain.warm_start(ext, str(path))  # strict load: weights fit despite 4x block_size
    for a, b in zip(ext.raw_model.parameters(), donor.raw_model.parameters(), strict=True):
        assert torch.equal(a, b)
    # a full-extended-length forward runs (the base model would assert at T > 16)
    x, y = _data(7).get_batch(NEW_BLOCK, 2, "cpu", torch.Generator().manual_seed(0))
    _, loss = ext.raw_model(x, y)
    assert loss.isfinite()


def test_extended_model_rope_cache_covers_new_block_at_raised_base(tmp_path):
    ext = _extended_trainer(tmp_path)
    head_dim = 32 // 2
    for block in ext.raw_model.transformer.h:
        assert block.attn.rope_cos.shape == (NEW_BLOCK, head_dim // 2)
        cos, sin = build_rope_cache(NEW_BLOCK, head_dim, base=NEW_BASE)
        assert torch.equal(block.attn.rope_cos, cos)
        assert torch.equal(block.attn.rope_sin, sin)


def test_causal_beyond_old_block(tmp_path):
    # Changing a token at position t must not change logits at positions < t, including
    # positions past the OLD context window.
    ext = _extended_trainer(tmp_path)
    model = ext.raw_model.eval()
    t = 50
    x = torch.randint(0, 64, (1, NEW_BLOCK), generator=torch.Generator().manual_seed(3))
    x2 = x.clone()
    x2[0, t] = (x2[0, t] + 1) % 64
    with torch.no_grad():
        la, _ = model(x)
        lb, _ = model(x2)
    assert torch.allclose(la[:, :t], lb[:, :t], atol=1e-5)
    assert not torch.allclose(la[:, t:], lb[:, t:], atol=1e-5)


def test_kv_cached_generation_runs_past_old_block(tmp_path):
    # The KV cache is allocated from config.block_size: capacity must follow the NEW
    # config, so generation crosses the old 16-token window instead of stopping there.
    ext = _extended_trainer(tmp_path)
    model = ext.raw_model
    idx = torch.randint(0, 64, (1, 20), generator=torch.Generator().manual_seed(5))
    out = generate_cached(model, idx, max_new_tokens=40, temperature=0.0)
    assert out.shape == (1, 60)  # 20 prompt + 40 generated, all past the old block


def test_load_for_eval_extends_frozen_checkpoint(tmp_path):
    # eval_passkey probes a SHORT-context checkpoint at long lengths: same weights, RoPE
    # cache extended, rope_base preserved from the checkpoint (NOT changed by the probe).
    donor = Trainer(_cfg(out_dir=str(tmp_path / "run")), _data())
    donor.train_step()
    donor.save_checkpoint(str(tmp_path / "run" / "ckpt_1.pt"))
    model, step, cfg, eval_block = ep.load_for_eval(tmp_path / "run", NEW_BLOCK, "cpu")
    assert step == 1 and cfg.block_size == OLD_BLOCK and eval_block == NEW_BLOCK
    assert model.config.block_size == NEW_BLOCK
    assert model.config.rope_base == OLD_BASE
    for a, b in zip(model.parameters(), donor.raw_model.parameters(), strict=True):
        assert torch.equal(a, b)
    out = generate_cached(model, torch.zeros((1, NEW_BLOCK - 4), dtype=torch.long), 4)
    assert out.shape == (1, NEW_BLOCK)


def test_load_for_eval_keeps_native_block_when_long_enough(tmp_path):
    donor = Trainer(_cfg(out_dir=str(tmp_path / "run")), _data())
    donor.save_checkpoint(str(tmp_path / "run" / "ckpt_0.pt"))
    model, _, _, eval_block = ep.load_for_eval(tmp_path / "run", 8, "cpu")
    assert eval_block == OLD_BLOCK and model.config.block_size == OLD_BLOCK
