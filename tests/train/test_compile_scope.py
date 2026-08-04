"""Per-block compile: the loss head stays out of the graph, checkpoints stay compatible.

Whole-model compile drags Liger's fused CE into the dynamo graph, and its
addmm(out_dtype=...) is untraceable on cu126 — it crashed every paid attempt and left
compile disabled entirely, at a measured 21.5% of step time. Compiling the blocks
individually gets the win (the blocks are where the bandwidth-bound elementwise work
lives) without ever tracing the loss path.
"""

from __future__ import annotations

import pytest
import torch

from microlab.train.config import RunConfig
from microlab.train.trainer import TensorData, Trainer


def _data():
    return TensorData(torch.randint(0, 64, (2000,),
                      generator=torch.Generator().manual_seed(7)))


def _tiny_cfg(**kw):
    return RunConfig(block_size=16, batch_size=1, n_layer=2, n_head=2, n_embd=32,
                     vocab_size=64, max_steps=1, device="cpu", **kw)


def test_default_scope_is_blocks():
    """The safe scope must be the default: "model" is the one that crashes on cu126,
    and a config that omits the field must not silently select it."""
    assert RunConfig(block_size=8, batch_size=1).compile_scope == "blocks"


def test_an_unknown_scope_raises_instead_of_silently_not_compiling():
    cfg = _tiny_cfg(compile=True)
    cfg.compile_scope = "everything"
    with pytest.raises(ValueError, match="compile_scope"):
        Trainer(cfg, _data())


def test_per_block_compile_keeps_state_dict_keys_identical():
    """THE resume property. nn.Module.compile() is in-place; the wrapper form
    (h[i] = torch.compile(blk)) returns an OptimizedModule whose keys grow an
    `_orig_mod.` prefix, so a checkpoint written by a compiled run would not load into
    an uncompiled model — resume would break exactly when a migrated box chose
    different compile settings."""
    plain = Trainer(_tiny_cfg(), _data())
    compiled = Trainer(_tiny_cfg(compile=True), _data())
    a = set(plain.raw_model.state_dict().keys())
    b = set(compiled.raw_model.state_dict().keys())
    assert a == b, f"compile changed state_dict keys: {sorted(b - a)[:4]}"
    assert not any("_orig_mod" in k for k in b)


def test_the_loss_head_is_not_compiled():
    """lm_head/ln_f must stay out of any dynamo graph — they feed Liger's fused CE."""
    t = Trainer(_tiny_cfg(compile=True), _data())
    m = t.raw_model
    for blk in m.transformer.h:
        assert hasattr(blk, "_compiled_call_impl") and blk._compiled_call_impl is not None
    assert getattr(m.lm_head, "_compiled_call_impl", None) is None
    assert getattr(m.transformer.ln_f, "_compiled_call_impl", None) is None


def test_a_compiled_step_still_runs_and_produces_a_finite_loss():
    t = Trainer(_tiny_cfg(compile=True), _data())
    loss = t.train_step()
    assert torch.isfinite(torch.tensor(loss)), f"loss {loss}"
