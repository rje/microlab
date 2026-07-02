"""Spec + validation for the hand-written Phase-5 interpretability primitives."""

import pytest
import torch

from microlab.exercises.phase05_interp import induction_score, logit_lens
from microlab.interp.reference import lens as ref
from microlab.model.reference.variants import VariantConfig, VariantGPT


def _model():
    torch.manual_seed(0)
    cfg = VariantConfig(vocab_size=64, block_size=64, n_layer=3, n_head=4, n_embd=32,
                        norm="rms", pos="rope", mlp="swiglu")
    return VariantGPT(cfg).eval()


def test_logit_lens_matches_reference_on_real_model():
    m = _model()
    x = torch.randint(0, 64, (2, 10))
    res = ref.collect_residual_stream(m, x)
    assert torch.allclose(
        logit_lens(res, m.transformer.ln_f, m.lm_head),
        ref.logit_lens(res, m.transformer.ln_f, m.lm_head), atol=1e-6,
    )


def test_induction_score_matches_reference():
    m = _model()
    g = torch.Generator().manual_seed(0)
    seq = ref.repeated_token_sequence(64, period=8, repeats=3, generator=g)
    attn = ref.attention_patterns(m, seq)
    assert torch.allclose(induction_score(attn, 8), ref.induction_score(attn, 8), atol=1e-6)

pytestmark = pytest.mark.exercise
