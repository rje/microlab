import math

import pytest
import torch

from microlab.model.reference.gpt import GPT, GPTConfig
from microlab.model.reference.reward import RewardModel, preference_loss, reward_accuracy


def test_preference_loss_known_values():
    # equal rewards -> -log sigmoid(0) = log 2
    assert preference_loss(torch.tensor([0.0]), torch.tensor([0.0])).item() == pytest.approx(
        math.log(2), abs=1e-5
    )
    # chosen >> rejected -> small loss
    assert preference_loss(torch.tensor([10.0]), torch.tensor([-10.0])).item() < 1e-3
    # chosen << rejected -> large loss
    assert preference_loss(torch.tensor([-10.0]), torch.tensor([10.0])).item() > 10.0


def test_reward_accuracy():
    c = torch.tensor([1.0, 2.0, -1.0])
    r = torch.tensor([0.0, 3.0, -2.0])
    assert reward_accuracy(c, r) == pytest.approx(2 / 3)


def test_reward_model_shapes():
    torch.manual_seed(0)
    rm = RewardModel(GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32)))
    idx = torch.randint(0, 64, (4, 16))
    assert rm(idx).shape == (4, 16)
    assert rm.sequence_reward(idx).shape == (4,)


@pytest.mark.gpu
def test_reward_model_learns_preference_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    torch.manual_seed(0)
    rm = RewardModel(
        GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=64))
    ).cuda()
    opt = torch.optim.AdamW(rm.parameters(), lr=1e-3)
    chosen = torch.randint(0, 64, (8, 16), device="cuda")
    rejected = torch.randint(0, 64, (8, 16), device="cuda")
    first = None
    for _ in range(100):
        loss = preference_loss(rm.sequence_reward(chosen), rm.sequence_reward(rejected))
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first  # it separates chosen from rejected
