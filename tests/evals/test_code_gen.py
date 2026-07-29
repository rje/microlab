"""generate_until: greedy output must match the reference generate_cached token-for-token
(minus the stop suffix), stop early on stop strings, and refuse prompts that cannot fit."""

from __future__ import annotations

import pytest
import torch

from microlab.evals.code.gen import generate_until
from microlab.infer.reference.kv_cache import generate_cached
from microlab.model.reference.variants import VariantConfig, VariantGPT
from microlab.tokenizer.fast import FastTokenizer


@pytest.fixture(scope="module")
def tok():
    corpus = ["the quick brown fox jumps over the lazy dog. " * 4,
              "def add(a, b):\n    return a + b\n", "hello world, hello tools. "]
    return FastTokenizer.train(corpus * 4, vocab_size=300)


@pytest.fixture(scope="module")
def model(tok):
    torch.manual_seed(0)
    cfg = VariantConfig(vocab_size=tok.vocab_size, block_size=128, n_layer=2, n_head=4,
                        n_embd=32, norm="rms", pos="rope", mlp="swiglu")
    return VariantGPT(cfg).eval()


def test_greedy_matches_generate_cached(model, tok):
    prompt_ids = tok.encode("the quick brown fox")
    ref = generate_cached(model, torch.tensor([prompt_ids]), 24, temperature=0.0)
    ref_text = tok.decode(ref[0].tolist()[len(prompt_ids):])
    out = generate_until(model, tok, prompt_ids, max_new=24, stops=[], device="cpu")
    assert out == ref_text


def test_stop_string_truncates_early(model, tok):
    prompt_ids = tok.encode("the quick brown fox")
    full = generate_until(model, tok, prompt_ids, max_new=24, stops=[], device="cpu")
    stop = full[5:9]
    out = generate_until(model, tok, prompt_ids, max_new=24, stops=[stop], device="cpu")
    assert out == full[: full.index(stop)]


def test_prompt_plus_budget_over_block_raises(model, tok):
    ids = tok.encode("hello world " * 40)[:120]
    with pytest.raises(ValueError, match="exceeds"):
        generate_until(model, tok, ids, max_new=64, stops=[], device="cpu")
