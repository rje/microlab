"""Interp oracle tests: residual-stream collection, logit lens, induction scoring."""

import pytest
import torch

from microlab.interp.reference.lens import (
    attention_patterns,
    collect_residual_stream,
    induction_score,
    logit_lens,
    repeated_token_sequence,
)
from microlab.model.reference.variants import VariantConfig, VariantGPT


def _model(n_kv_head=None):
    torch.manual_seed(0)
    cfg = VariantConfig(vocab_size=64, block_size=64, n_layer=3, n_head=4, n_embd=32,
                        norm="rms", pos="rope", mlp="swiglu", n_kv_head=n_kv_head)
    return VariantGPT(cfg).eval()


@pytest.mark.parametrize("n_kv_head", [None, 2])
def test_residual_stream_shapes_and_final_equals_forward(n_kv_head):
    m = _model(n_kv_head)
    x = torch.randint(0, 64, (2, 10))
    res = collect_residual_stream(m, x)
    assert len(res) == 4 and all(r.shape == (2, 10, 32) for r in res)
    logits, _ = m(x)
    lens = logit_lens(res, m.transformer.ln_f, m.lm_head)
    assert lens.shape == (4, 2, 10, 64)
    assert torch.allclose(lens[-1], logits, atol=1e-5)  # last layer IS the model output


@pytest.mark.parametrize("n_kv_head", [None, 2])
def test_attention_patterns_rows_sum_to_one_and_causal(n_kv_head):
    m = _model(n_kv_head)
    x = torch.randint(0, 64, (1, 12))
    attn = attention_patterns(m, x)
    assert attn.shape == (3, 4, 12, 12)
    assert torch.allclose(attn.sum(-1), torch.ones(3, 4, 12), atol=1e-5)
    assert torch.all(torch.triu(attn, diagonal=1).abs() < 1e-6)  # no attention to future


def test_induction_score_perfect_head_is_one():
    T, P = 12, 4
    attn = torch.zeros(1, T, T)
    for i in range(P, T):
        attn[0, i, i - P + 1] = 1.0  # a textbook induction head
    assert torch.allclose(induction_score(attn, P), torch.ones(1), atol=1e-6)


def test_repeated_sequence_repeats():
    g = torch.Generator().manual_seed(0)
    seq = repeated_token_sequence(64, period=8, repeats=3, generator=g)
    assert seq.shape == (1, 24)
    assert torch.equal(seq[0, :8], seq[0, 8:16]) and torch.equal(seq[0, :8], seq[0, 16:])
