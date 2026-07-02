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

pytestmark = pytest.mark.exercise
