"""KV-cache oracle: cached generation must EXACTLY match uncached generation, and be
usable in prefill + one-token-step decoding."""

import pytest
import torch

from microlab.infer.reference.kv_cache import KVCache, generate_cached
from microlab.model.reference.sample import generate
from microlab.model.reference.variants import VariantConfig, VariantGPT


def _model(n_kv_head=None):
    torch.manual_seed(0)
    cfg = VariantConfig(vocab_size=64, block_size=64, n_layer=3, n_head=4, n_embd=32,
                        norm="rms", pos="rope", mlp="swiglu", n_kv_head=n_kv_head)
    return VariantGPT(cfg).eval()


@pytest.mark.parametrize("n_kv_head", [None, 2])
def test_cached_forward_matches_uncached(n_kv_head):
    m = _model(n_kv_head)
    x = torch.randint(0, 64, (2, 10))
    full_logits, _ = m(x)
    cache = KVCache(3, 2, n_kv_head or 4, 64, 8)
    pre_logits, _ = m(x[:, :6], kv_cache=cache)          # prefill
    assert torch.allclose(pre_logits, full_logits[:, :6], atol=1e-5)
    for t in range(6, 10):                                # one-token steps
        step_logits, _ = m(x[:, t:t + 1], kv_cache=cache)
        assert torch.allclose(step_logits[:, 0], full_logits[:, t], atol=1e-4)


def test_generate_cached_exactly_matches_reference_greedy():
    m = _model()
    idx = torch.randint(0, 64, (2, 8))
    assert torch.equal(
        generate_cached(m, idx.clone(), 20, temperature=0.0),
        generate(m, idx.clone(), 20, temperature=0.0),
    )


def test_default_path_untouched():
    m = _model()
    x = torch.randint(0, 64, (2, 10))
    torch.manual_seed(1)
    a, _ = m(x)
    torch.manual_seed(1)
    b, _ = m(x, kv_cache=None)
    assert torch.equal(a, b)
