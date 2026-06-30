import torch

from microlab.model.reference.gpt import GPT, GPTConfig
from microlab.model.reference.sample import generate


def _m():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=32, block_size=16, n_layer=2, n_head=2, n_embd=32))


def test_generate_grows_by_max_new_tokens():
    m = _m()
    idx = torch.zeros((1, 1), dtype=torch.long)
    out = generate(m, idx, max_new_tokens=10)
    assert out.shape == (1, 11)


def test_greedy_is_deterministic():
    m = _m()
    idx = torch.zeros((1, 1), dtype=torch.long)
    a = generate(m, idx, max_new_tokens=8, temperature=0.0)
    b = generate(m, idx, max_new_tokens=8, temperature=0.0)
    assert torch.equal(a, b)


def test_generation_crops_to_block_size():
    m = _m()  # block_size 16
    idx = torch.zeros((1, 20), dtype=torch.long)  # already longer than block
    out = generate(m, idx, max_new_tokens=5, temperature=0.0)
    assert out.shape == (1, 25)  # no assertion error from the forward block-size check
