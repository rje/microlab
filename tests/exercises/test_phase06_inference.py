"""Spec + validation for the hand-written Phase-6 inference primitives."""

import time

import pytest
import torch

from microlab.model.reference.sample import generate
from microlab.model.reference.variants import VariantConfig, VariantGPT


def _model(block_size=256):
    torch.manual_seed(0)
    cfg = VariantConfig(vocab_size=64, block_size=block_size, n_layer=4, n_head=4,
                        n_embd=64, norm="rms", pos="rope", mlp="swiglu")
    return VariantGPT(cfg).eval()


def test_generate_cached_exact_match_and_faster():
    from microlab.exercises.phase06_inference import generate_cached
    m = _model()
    idx = torch.randint(0, 64, (1, 8))
    t0 = time.perf_counter()
    ref = generate(m, idx.clone(), 200, temperature=0.0)
    t_ref = time.perf_counter() - t0
    t0 = time.perf_counter()
    stu = generate_cached(m, idx.clone(), 200, temperature=0.0)
    t_stu = time.perf_counter() - t0
    assert torch.equal(stu, ref), "cached generation must be token-for-token identical"
    assert t_stu < t_ref, f"cache should be faster: cached={t_stu:.3f}s uncached={t_ref:.3f}s"


def test_sample_next_matches_reference():
    from microlab.exercises.phase06_inference import sample_next
    from microlab.infer.reference.sampling import sample_next as ref_sample
    torch.manual_seed(0)
    logits = torch.randn(4, 32)
    for kwargs in [dict(temperature=0.0), dict(temperature=0.8, top_k=5),
                   dict(temperature=1.0, top_p=0.9), dict(temperature=0.7, top_k=8, top_p=0.95)]:
        a = sample_next(logits.clone(), generator=torch.Generator().manual_seed(3), **kwargs)
        b = ref_sample(logits.clone(), generator=torch.Generator().manual_seed(3), **kwargs)
        assert torch.equal(a, b), kwargs


def test_quantize_groupwise_matches_reference():
    from microlab.exercises.phase06_inference import quantize_groupwise
    from microlab.infer.reference.quant import quantize_groupwise as ref_q
    torch.manual_seed(0)
    w = torch.randn(32, 128)
    for bits in (4, 8):
        assert torch.allclose(quantize_groupwise(w, bits=bits), ref_q(w, bits=bits), atol=1e-6)


def test_speculative_accept_matches_reference():
    from microlab.exercises.phase06_inference import speculative_accept
    from microlab.infer.reference.speculative import speculative_accept as ref_acc
    torch.manual_seed(0)
    for seed in range(10):
        draft = torch.softmax(torch.randn(4, 16), -1)
        target = torch.softmax(torch.randn(4, 16), -1)
        tokens = torch.multinomial(draft, 1).squeeze(1)
        a = speculative_accept(tokens, draft, target, torch.Generator().manual_seed(seed))
        b = ref_acc(tokens, draft, target, torch.Generator().manual_seed(seed))
        assert a[0] == b[0]
        assert (a[1] is None) == (b[1] is None)
        if a[1] is not None:
            assert torch.equal(a[1], b[1])

pytestmark = pytest.mark.exercise
