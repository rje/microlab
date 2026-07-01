import pytest
import torch

from microlab.model.reference.gpt import GPT, GPTConfig
from microlab.model.reference.train import TrainConfig, estimate_loss, overfit_batch, train


def test_estimate_loss_returns_positive_float():
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=32, block_size=16, n_layer=2, n_head=2, n_embd=32))
    data = torch.randint(0, 32, (2000,))
    loss = estimate_loss(m, data, block_size=16, batch_size=8, iters=5, device="cpu")
    assert isinstance(loss, float) and loss > 0


def test_estimate_loss_leaves_model_in_train_mode():
    m = GPT(GPTConfig(vocab_size=32, block_size=16, n_layer=2, n_head=2, n_embd=32))
    m.train()
    estimate_loss(m, torch.randint(0, 32, (500,)), 16, 4, iters=2)
    assert m.training  # eval mode is restored


def test_train_reports_val_loss_when_val_data_given():
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=32, block_size=16, n_layer=2, n_head=2, n_embd=32))
    data = torch.randint(0, 32, (3000,))
    val = torch.randint(0, 32, (1000,))
    cfg = TrainConfig(steps=10, batch_size=8, block_size=16, device="cpu")
    stats = train(m, data, cfg, val_data=val)
    assert isinstance(stats["val_loss"], float) and stats["val_loss"] > 0
    # no val_data -> None
    stats2 = train(m, data, TrainConfig(steps=5, batch_size=8, block_size=16, device="cpu"))
    assert stats2["val_loss"] is None


def test_overfit_single_batch_collapses_loss_cpu():
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=32, block_size=16, n_layer=2, n_head=2, n_embd=32))
    x = torch.randint(0, 32, (4, 16))
    y = torch.randint(0, 32, (4, 16))
    losses = overfit_batch(m, x, y, steps=300, lr=1e-3, device="cpu")
    assert losses[-1] < losses[0] * 0.2  # memorized the batch


@pytest.mark.gpu
def test_train_runs_on_cuda_and_loss_drops():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=2, n_embd=64))
    data = torch.randint(0, 64, (5000,))
    stats = train(m, data, TrainConfig(steps=60, batch_size=16, block_size=32,
                                       device="cuda", log_every=1000))
    assert stats["device"] == "cuda"
    assert stats["peak_vram_mb"] > 0 and stats["tokens_per_sec"] > 0
    assert stats["history"][-1] < stats["history"][0]
