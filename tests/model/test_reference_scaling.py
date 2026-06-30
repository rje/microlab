import math

import pytest
import torch

from microlab.model.reference.gpt import GPT, GPTConfig
from microlab.model.reference.scaling import (
    count_params,
    fit_scaling_law,
    model_family,
    training_flops,
    training_flops_per_token,
)


@pytest.mark.parametrize(
    "cfg",
    [
        GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32),
        GPTConfig(vocab_size=512, block_size=128, n_layer=4, n_head=4, n_embd=192),
        GPTConfig(vocab_size=100, block_size=64, n_layer=3, n_head=3, n_embd=96, bias=False),
    ],
)
def test_count_params_matches_real_model(cfg):
    assert count_params(cfg)["total"] == GPT(cfg).num_params()


def test_flops_is_six_times_nonembedding():
    cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32)
    n = count_params(cfg)["non_embedding"]
    assert training_flops_per_token(cfg) == 6 * n
    assert training_flops(cfg, 1000) == 6 * n * 1000


def test_fit_recovers_known_power_law():
    A_true, alpha_true = 3.0, 0.1
    params = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    losses = [A_true * n ** (-alpha_true) for n in params]
    A, alpha = fit_scaling_law(params, losses)
    assert A == pytest.approx(A_true, rel=1e-3)
    assert alpha == pytest.approx(alpha_true, rel=1e-3)


def test_model_family_increases_in_size():
    cfgs = model_family([64, 128, 256])
    sizes = [GPT(c).num_params() for c in cfgs]
    assert sizes == sorted(sizes) and len(set(sizes)) == 3


@pytest.mark.gpu
def test_scaling_sweep_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from microlab.model.reference.scaling import run_scaling_sweep
    from microlab.model.reference.train import TrainConfig

    data = torch.randint(0, 512, (8000,))
    res = run_scaling_sweep(data, [64, 128], TrainConfig(steps=40, batch_size=16,
                                                         block_size=64, device="cuda"))
    assert len(res["points"]) == 2
    assert all(p["params"] > 0 and p["loss"] > 0 for p in res["points"])
    assert math.isfinite(res["alpha"])
