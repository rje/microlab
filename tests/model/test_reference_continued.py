import pytest
import torch

from microlab.model.reference.continued import (
    build_replay_mix,
    continued_pretrain,
    evaluate_on_corpora,
    forgetting_score,
    interpolated_rope_cache,
)
from microlab.model.reference.gpt import GPT, GPTConfig
from microlab.model.reference.train import TrainConfig
from microlab.model.reference.variants import build_rope_cache


def test_forgetting_score_sign():
    assert forgetting_score(3.0, 3.5) == pytest.approx(0.5)   # forgot
    assert forgetting_score(3.0, 3.0) == pytest.approx(0.0)   # retained
    assert forgetting_score(3.0, 2.7) == pytest.approx(-0.3)  # improved


def test_replay_mix_zero_is_identity():
    new = torch.arange(100)
    assert torch.equal(build_replay_mix(new, torch.arange(50), 0.0), new)


def test_replay_mix_fraction_is_correct():
    new = torch.arange(900)
    old = torch.arange(10000, 20000)
    mixed = build_replay_mix(new, old, 0.1)  # want ~10% old
    n_old = (mixed >= 10000).sum().item()
    assert n_old / len(mixed) == pytest.approx(0.1, abs=0.01)
    assert len(mixed) == 900 + n_old


def test_replay_mix_caps_at_available_old():
    new = torch.arange(1000)
    old = torch.arange(10000, 10005)  # only 5 old tokens available
    mixed = build_replay_mix(new, old, 0.9)  # asks for lots, only 5 exist
    assert len(mixed) == 1005


def test_replay_mix_rejects_fraction_one():
    with pytest.raises(AssertionError):
        build_replay_mix(torch.arange(10), torch.arange(10), 1.0)


def test_evaluate_on_corpora_returns_float_per_corpus():
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=32, block_size=16, n_layer=2, n_head=2, n_embd=32))
    corpora = {"a": torch.randint(0, 32, (2000,)), "b": torch.randint(0, 32, (2000,))}
    res = evaluate_on_corpora(m, corpora, 16, 8, iters=3, device="cpu")
    assert set(res) == {"a", "b"} and all(isinstance(v, float) and v > 0 for v in res.values())


@pytest.mark.gpu
def test_continued_pretrain_reports_forgetting_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=2, n_embd=64))
    old = torch.randint(0, 64, (6000,))
    new = torch.randint(0, 64, (6000,))
    res = continued_pretrain(
        m, new, {"old": old, "new": new},
        TrainConfig(steps=50, batch_size=16, block_size=32, device="cuda"),
    )
    assert set(res["forgetting"]) == {"old", "new"}
    assert all(isinstance(v, float) for v in res["forgetting"].values())
    assert res["train"]["device"] == "cuda"


# Position-interpolation oracle: scaled positions must land exactly on the original
# cache's rows at integer-aligned points.
def test_scale_one_is_identity():
    a_cos, a_sin = build_rope_cache(64, 8)
    b_cos, b_sin = interpolated_rope_cache(64, 8, scale=1.0)
    assert torch.allclose(a_cos, b_cos, atol=1e-6) and torch.allclose(a_sin, b_sin, atol=1e-6)


def test_scale_two_hits_original_positions_at_even_rows():
    base_cos, base_sin = build_rope_cache(32, 8)
    int_cos, int_sin = interpolated_rope_cache(64, 8, scale=2.0)
    assert torch.allclose(int_cos[::2], base_cos, atol=1e-5)
    assert torch.allclose(int_sin[::2], base_sin, atol=1e-5)


def test_interpolated_frequencies_stay_in_trained_range():
    cos, _ = interpolated_rope_cache(128, 8, scale=4.0)
    base_cos, _ = build_rope_cache(32, 8)
    assert cos.shape[0] == 128
    # scale=4 -> interpolated row 124 sits at position 124/4 == 31, the last trained row;
    # its rotation angles equal the last row of the 32-length base cache exactly.
    assert torch.allclose(cos[124], base_cos[31], atol=1e-5)
