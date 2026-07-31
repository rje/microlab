"""QK-norm on the global-attention layers (Qwen3/Gemma-3 head_dim variant)."""
import torch

from microlab.model.reference.variants import MLAAttention, VariantConfig, VariantGPT


def _cfg(**kw):
    base = dict(vocab_size=512, block_size=128, n_layer=8, n_head=4, n_embd=256,
                dropout=0.0, norm="rms", pos="nope", mlp="swiglu", block_norm="peri",
                hybrid_every=4, global_attn="mla", mla_kv_lora=64, gdn_gate="channel")
    base.update(kw)
    return VariantConfig(**base)


def test_normalises_over_head_dim_not_the_full_projection():
    """The two published variants differ: OLMo 2 normalises over n_head*head_dim, Qwen3 and
    Gemma 3 over head_dim. Getting this wrong silently couples the heads."""
    a = MLAAttention(_cfg(qk_norm=True))
    assert a.q_norm.weight.shape == (a.head_dim,), a.q_norm.weight.shape
    assert a.k_norm.weight.shape == (a.head_dim,), a.k_norm.weight.shape


def test_off_by_default_leaves_the_param_tree_untouched():
    off = VariantGPT(_cfg(qk_norm=False))
    on = VariantGPT(_cfg(qk_norm=True))
    extra = set(dict(on.named_parameters())) - set(dict(off.named_parameters()))
    assert all(n.endswith(("q_norm.weight", "k_norm.weight")) for n in extra), extra
    # only the global layers gain norms: 2 per MLA layer, 2 MLA layers in an 8-layer 3:1 stack
    assert len(extra) == 4, extra


def test_changes_the_output():
    """Guard against the norm being constructed but never applied."""
    torch.manual_seed(0)
    a = MLAAttention(_cfg(qk_norm=True)).eval()
    x = torch.randn(1, 32, 256)
    with_norm = a(x)
    a.qk_norm = False
    assert not torch.allclose(with_norm, a(x), atol=1e-5)


def test_forward_backward_finite():
    torch.manual_seed(0)
    m = VariantGPT(_cfg(qk_norm=True))
    idx = torch.randint(0, 512, (2, 128))
    _, loss = m(idx, targets=idx)
    loss.backward()
    g = m.transformer.h[3].attn
    assert torch.isfinite(g.q_norm.weight.grad).all()
    assert torch.isfinite(g.k_norm.weight.grad).all()


def test_every_variant_field_is_reachable_from_runconfig():
    """REGRESSION GUARD. The 1B shipped MHA because RunConfig never exposed n_kv_head
    (docs/sota-parity-1b.md #8); the same gap then blocked gdn_gate/global_attn/qk_norm.
    An architecture knob the config cannot set is a knob that silently defaults."""
    from dataclasses import fields

    from microlab.model.reference.variants import VariantConfig
    from microlab.train.config import RunConfig

    arch = {"n_kv_head", "rope_base", "block_norm", "hybrid_every", "gdn_chunk",
            "gdn_conv_kernel", "gdn_fused", "gdn_gate", "global_attn", "mla_kv_lora",
            "qk_norm"}
    run = {f.name for f in fields(RunConfig)}
    var = {f.name for f in fields(VariantConfig)}
    missing_in_run = arch - run
    missing_in_var = arch - var
    assert not missing_in_run, f"not reachable from RunConfig: {missing_in_run}"
    assert not missing_in_var, f"not on VariantConfig: {missing_in_var}"
