"""RunConfig -> Trainer -> VariantConfig plumbing for n_kv_head and rope_base
(sota-parity finding #8: the reference track had GQA/RoPE-base, the production config
surface didn't). Defaults must reproduce current behavior bit-identically so every
existing config and checkpoint is unaffected."""

import pickle

import torch

from microlab.model.reference.variants import (
    GQAAttention,
    RoPECausalSelfAttention,
    VariantConfig,
    VariantGPT,
    build_rope_cache,
)
from microlab.train.config import RunConfig
from microlab.train.trainer import TensorData, Trainer


def _cfg(**kw):
    base = dict(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32,
                warmup_steps=2, max_steps=4, lr_decay_steps=4, batch_size=4,
                eval_interval=1000, ckpt_interval=1000, device="cpu", dtype="float32")
    base.update(kw)
    return RunConfig(**base)


def _data():
    return TensorData(torch.randint(0, 64, (2000,), generator=torch.Generator().manual_seed(7)))


def test_defaults_reproduce_old_model_bit_identically():
    # A default-RunConfig Trainer must build the exact model the pre-knob Trainer built:
    # same module types, same weights (same seed => same init RNG stream), same logits.
    cfg = _cfg()
    tr = Trainer(cfg, _data())
    torch.manual_seed(cfg.seed)
    old = VariantGPT(VariantConfig(
        vocab_size=cfg.vocab_size, block_size=cfg.block_size, n_layer=cfg.n_layer,
        n_head=cfg.n_head, n_embd=cfg.n_embd, dropout=cfg.dropout, norm=cfg.norm,
        pos=cfg.pos, mlp=cfg.mlp,
    ))
    assert isinstance(tr.raw_model.transformer.h[0].attn, RoPECausalSelfAttention)
    x = torch.randint(0, 64, (2, 8), generator=torch.Generator().manual_seed(0))
    la, _ = tr.raw_model(x)
    lb, _ = old(x)
    assert torch.equal(la, lb)
    assert list(tr.raw_model.state_dict().keys()) == list(old.state_dict().keys())


def test_trainer_builds_gqa_model():
    tr = Trainer(_cfg(n_kv_head=1), _data())
    attn = tr.raw_model.transformer.h[0].attn
    assert isinstance(attn, GQAAttention)
    assert attn.n_kv_head == 1


def test_trainer_rope_base_reaches_cache():
    tr = Trainer(_cfg(rope_base=500000.0), _data())
    attn = tr.raw_model.transformer.h[0].attn
    cos, _ = build_rope_cache(16, 16, base=500000.0)
    assert torch.equal(attn.rope_cos, cos)


def test_checkpoint_roundtrips_new_fields(tmp_path):
    cfg = _cfg(n_kv_head=2, rope_base=500000.0, out_dir=str(tmp_path))
    tr = Trainer(cfg, _data())
    tr.train()
    path = tmp_path / "ckpt_4.pt"
    tr.save_checkpoint(str(path))
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    assert ckpt["cfg"].n_kv_head == 2
    assert ckpt["cfg"].rope_base == 500000.0
    # And a fresh Trainer built from that cfg loads it (GQA-shaped state dict round-trip).
    tr2 = Trainer(ckpt["cfg"], _data())
    tr2.load_checkpoint(str(path))
    for a, b in zip(tr.raw_model.parameters(), tr2.raw_model.parameters(), strict=True):
        assert torch.equal(a, b)


def test_old_pickled_cfg_gains_defaults():
    # Checkpoints saved before these fields existed unpickle without them in __dict__;
    # attribute access must fall back to the dataclass class-level defaults (MHA, 10000.0).
    cfg = _cfg()
    cfg.__dict__.pop("n_kv_head")
    cfg.__dict__.pop("rope_base")
    old = pickle.loads(pickle.dumps(cfg))
    assert old.n_kv_head is None
    assert old.rope_base == 10000.0
