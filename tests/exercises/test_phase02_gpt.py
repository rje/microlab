"""Spec + validation for the hand-written Phase-2 transformer pieces.

Implement ``microlab.exercises.phase02_gpt`` until these pass. Each differential test copies
the reference oracle's weights into your module and asserts identical outputs, so green
means your math matches the oracle (not just "plausible").
"""

import pytest
import torch

from microlab.exercises.phase02_gpt import (
    StudentBlock,
    StudentCausalSelfAttention,
    generate,
    train_step,
)
from microlab.model.reference.gpt import GPT, Block, CausalSelfAttention, GPTConfig
from microlab.model.reference.sample import generate as ref_generate


def _cfg():
    return GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32)


def test_attention_matches_reference():
    torch.manual_seed(0)
    cfg = _cfg()
    ref = CausalSelfAttention(cfg).eval()
    stu = StudentCausalSelfAttention(cfg).eval()
    stu.load_state_dict(ref.state_dict())
    x = torch.randn(2, 16, 32)
    assert torch.allclose(stu(x), ref(x), atol=1e-4)


def test_attention_is_causal():
    torch.manual_seed(0)
    stu = StudentCausalSelfAttention(_cfg()).eval()
    x = torch.randn(1, 16, 32)
    a = stu(x)
    x2 = x.clone()
    x2[0, -1] += 1.0
    b = stu(x2)
    # position t must not attend to the changed final token
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)


def test_block_matches_reference():
    torch.manual_seed(0)
    cfg = _cfg()
    ref = Block(cfg).eval()
    stu = StudentBlock(cfg).eval()
    stu.load_state_dict(ref.state_dict())
    x = torch.randn(2, 16, 32)
    assert torch.allclose(stu(x), ref(x), atol=1e-4)


def test_train_step_overfits_a_batch():
    torch.manual_seed(0)
    model = GPT(_cfg())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 64, (4, 16))
    y = torch.randint(0, 64, (4, 16))
    first = train_step(model, x, y, opt)
    last = first
    for _ in range(200):
        last = train_step(model, x, y, opt)
    assert isinstance(last, float)
    assert last < first * 0.2  # the loop actually learns the batch


def test_generate_matches_reference_greedy():
    torch.manual_seed(0)
    model = GPT(_cfg())
    idx = torch.zeros((1, 1), dtype=torch.long)
    mine = generate(model, idx, max_new_tokens=10, temperature=0.0)
    ref = ref_generate(model, idx, max_new_tokens=10, temperature=0.0)
    assert torch.equal(mine, ref)

pytestmark = pytest.mark.exercise
