"""Rebuilding a VariantConfig from a checkpoint must not lose architecture fields.

This is a regression guard for a bug that has now happened twice. Both checkpoint loaders
(`load_variant_from_run` and `eval_passkey.load_for_eval`) listed the fields to copy BY
HAND, so every new architecture field silently failed to propagate:

  - round one: `block_norm` and `hybrid_every` missing, so no Peri-LN or hybrid run
    could be loaded;
  - round two: all five frontier fields missing (`gdn_gate`, `global_attn`,
    `mla_kv_lora`, `qk_norm`, `gdn_fused`), so the finished 32k frontier checkpoint
    could not be loaded by ANY eval script — discovered only when Phase C tried to run
    and hit `size mismatch for transformer.h.4.attn.a_proj.weight: [768, 768] vs
    [12, 768]`, a shape error a long way from its cause.

`variant_config_from_ckpt` now enumerates the dataclass instead. The test below is what
keeps that honest: it fails the moment a field can be dropped in transit, whatever the
field is, without anyone remembering to update a list.
"""

from __future__ import annotations

import dataclasses

import pytest

from microlab.model.reference.checkpoint import variant_config_from_ckpt
from microlab.model.reference.variants import VariantConfig

# A value differing from the dataclass default for EVERY field. If a new field is added
# without a line here, `test_every_field_is_covered_by_this_test` fails and points at it.
NON_DEFAULT = {
    "vocab_size": 256, "block_size": 64, "n_layer": 2, "n_head": 2, "n_embd": 64,
    "dropout": 0.1, "bias": False, "norm": "rms", "pos": "nope", "mlp": "swiglu",
    "block_norm": "peri", "n_kv_head": 1, "rope_base": 50000.0, "hybrid_every": 2,
    "gdn_chunk": 32, "gdn_conv_kernel": 3, "gdn_fused": False, "gdn_gate": "channel",
    "global_attn": "mla", "mla_kv_lora": 32, "qk_norm": True,
}


def test_every_field_is_covered_by_this_test():
    """Adding an architecture field must force a decision about propagating it."""
    fields = {f.name for f in dataclasses.fields(VariantConfig)}
    assert fields == set(NON_DEFAULT), (
        f"NON_DEFAULT is out of sync with VariantConfig: "
        f"missing {fields - set(NON_DEFAULT)}, stale {set(NON_DEFAULT) - fields}")


def test_variant_config_round_trips_every_field():
    """The bug in one assertion: every non-default value must survive the rebuild."""
    stored = VariantConfig(**NON_DEFAULT)
    rebuilt = variant_config_from_ckpt(stored)
    for name, want in NON_DEFAULT.items():
        if name == "dropout":
            continue                    # deliberately forced to 0.0 for inference
        assert getattr(rebuilt, name) == want, f"{name} lost in rebuild"


def test_dropout_is_forced_off_for_inference():
    rebuilt = variant_config_from_ckpt(VariantConfig(**NON_DEFAULT))
    assert rebuilt.dropout == 0.0


def test_overrides_win():
    """eval_passkey extends block_size past the trained window; nothing else changes."""
    stored = VariantConfig(**NON_DEFAULT)
    rebuilt = variant_config_from_ckpt(stored, block_size=4096)
    assert rebuilt.block_size == 4096
    assert rebuilt.gdn_gate == "channel"      # untouched by the override
    assert rebuilt.global_attn == "mla"


def test_missing_fields_fall_back_to_era_defaults():
    """A checkpoint predating a field must rebuild as that era's model, not crash."""

    class OldCfg:
        vocab_size, block_size, n_layer, n_head, n_embd = 256, 64, 2, 2, 64
        norm, pos, mlp = "rms", "rope", "swiglu"

    rebuilt = variant_config_from_ckpt(OldCfg())
    assert rebuilt.block_norm == "pre"        # pre-Peri-LN era
    assert rebuilt.hybrid_every is None       # dense
    assert rebuilt.gdn_gate == "scalar"
    assert rebuilt.global_attn == "gqa"
    assert rebuilt.qk_norm is False


@pytest.mark.parametrize("field", ["gdn_gate", "global_attn", "mla_kv_lora", "qk_norm"])
def test_frontier_fields_specifically(field):
    """Named explicitly because these are four of the five that broke Phase C."""
    rebuilt = variant_config_from_ckpt(VariantConfig(**NON_DEFAULT))
    assert getattr(rebuilt, field) == NON_DEFAULT[field]
