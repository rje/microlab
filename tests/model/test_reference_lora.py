import pytest
import torch
import torch.nn as nn

from microlab.model.reference.gpt import GPT, GPTConfig
from microlab.model.reference.lora import (
    LoRALinear,
    apply_lora_to_gpt,
    count_total,
    count_trainable,
    quantize_dequantize,
)
from microlab.model.reference.train import TrainConfig, train


def test_lora_is_noop_at_init():
    torch.manual_seed(0)
    base = nn.Linear(16, 24)
    lora = LoRALinear(base, rank=4, alpha=8).eval()
    x = torch.randn(3, 16)
    assert torch.allclose(lora(x), base(x), atol=1e-6)  # B=0 -> identity


def test_lora_merge_matches_forward():
    torch.manual_seed(0)
    base = nn.Linear(16, 24)
    lora = LoRALinear(base, rank=4, alpha=8)
    lora.lora_B.data.normal_()  # make the adapter nonzero
    lora.eval()
    merged = lora.merge().eval()
    x = torch.randn(5, 16)
    assert torch.allclose(lora(x), merged(x), atol=1e-5)


def test_lora_only_adapters_trainable():
    base = nn.Linear(16, 24)
    lora = LoRALinear(base, rank=4, alpha=8)
    trainable = {n for n, p in lora.named_parameters() if p.requires_grad}
    assert trainable == {"lora_A", "lora_B"}  # base frozen


def test_apply_lora_freezes_base_and_trains_little():
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=64))
    total_before = count_total(m)
    apply_lora_to_gpt(m, rank=8, alpha=16)
    frac = count_trainable(m) / total_before
    assert 0.0 < frac < 0.2  # adapters are a small fraction of the model


def test_quantize_dequantize_shape_and_bits():
    torch.manual_seed(0)
    w = torch.randn(64, 32)
    dq4 = quantize_dequantize(w, bits=4)
    dq8 = quantize_dequantize(w, bits=8)
    assert dq4.shape == w.shape
    err4 = (dq4 - w).abs().mean().item()
    err8 = (dq8 - w).abs().mean().item()
    assert err8 < err4  # more bits -> closer reconstruction
    assert err4 < w.abs().mean().item()  # still better than zero


@pytest.mark.gpu
def test_lora_training_reduces_loss_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=2, n_embd=64))
    apply_lora_to_gpt(m, rank=8, alpha=16)
    data = torch.randint(0, 64, (5000,))
    stats = train(m, data, TrainConfig(steps=60, batch_size=16, block_size=32, device="cuda"))
    assert stats["device"] == "cuda"
    assert stats["history"][-1] < stats["history"][0]  # adapters learn
