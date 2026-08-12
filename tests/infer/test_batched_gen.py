"""generate_batch: every greedy row must match generate_until token-for-token on both a
dense and a hybrid (KDA/MLA) model, stops must truncate per row, sampling must be
deterministic for a fixed (seed, n), and the context guard must refuse oversize work."""

from __future__ import annotations

import pytest
import torch

from microlab.evals.code.gen import generate_until
from microlab.infer.batched import generate_batch
from microlab.model.reference.variants import VariantConfig, VariantGPT
from microlab.tokenizer.fast import FastTokenizer


@pytest.fixture(scope="module")
def tok():
    corpus = ["the quick brown fox jumps over the lazy dog. " * 4,
              "def add(a, b):\n    return a + b\n", "hello world, hello tools. "]
    return FastTokenizer.train(corpus * 4, vocab_size=300)


@pytest.fixture(scope="module")
def dense(tok):
    torch.manual_seed(0)
    cfg = VariantConfig(vocab_size=tok.vocab_size, block_size=128, n_layer=2, n_head=4,
                        n_embd=32, norm="rms", pos="rope", mlp="swiglu")
    return VariantGPT(cfg).eval()


@pytest.fixture(scope="module")
def hybrid(tok):
    torch.manual_seed(0)
    cfg = VariantConfig(vocab_size=tok.vocab_size, block_size=128, n_layer=4, n_head=4,
                        n_embd=32, norm="rms", pos="nope", mlp="swiglu",
                        hybrid_every=4, global_attn="mla", mla_kv_lora=16,
                        gdn_gate="channel")
    return VariantGPT(cfg).eval()


@pytest.mark.parametrize("which", ["dense", "hybrid"])
def test_greedy_rows_match_generate_until(which, dense, hybrid, tok):
    model = {"dense": dense, "hybrid": hybrid}[which]
    prompt_ids = tok.encode("the quick brown fox")
    ref = generate_until(model, tok, prompt_ids, max_new=24, stops=[], device="cpu")
    outs = generate_batch(model, tok, prompt_ids, n=3, max_new=24, stops=[],
                          device="cpu")
    assert outs == [ref, ref, ref]


def test_stop_truncates_per_row(dense, tok):
    prompt_ids = tok.encode("the quick brown fox")
    full = generate_until(dense, tok, prompt_ids, max_new=24, stops=[], device="cpu")
    stop = full[5:9]
    outs = generate_batch(dense, tok, prompt_ids, n=2, max_new=24, stops=[stop],
                          device="cpu")
    assert outs == [full[: full.index(stop)]] * 2


def test_sampled_deterministic_and_rows_differ(dense, tok):
    prompt_ids = tok.encode("the quick brown fox")

    def run():
        gen = torch.Generator().manual_seed(7)
        return generate_batch(dense, tok, prompt_ids, n=4, max_new=16, stops=[],
                              device="cpu", temperature=1.0, top_k=50, generator=gen)

    a, b = run(), run()
    assert a == b
    assert len(set(a)) > 1, "shared-generator rows should decorrelate"


def test_oversize_prompt_raises(dense, tok):
    ids = tok.encode("hello world " * 40)[:120]
    with pytest.raises(ValueError, match="exceeds"):
        generate_batch(dense, tok, ids, n=2, max_new=64, stops=[], device="cpu")


def test_n_below_one_raises(dense, tok):
    with pytest.raises(ValueError, match="n must be"):
        generate_batch(dense, tok, tok.encode("hi"), n=0, max_new=8, stops=[],
                       device="cpu")
