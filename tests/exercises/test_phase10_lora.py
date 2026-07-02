"""Spec + validation for the hand-written Phase-10 LoRA / QLoRA primitives.

Implement ``microlab.exercises.phase10_lora`` until these pass. Differential tests grade you
against ``microlab.model.reference.lora``.
"""

import pytest
import torch
import torch.nn as nn

from microlab.exercises.phase10_lora import LoRALinear, quantize_dequantize
from microlab.model.reference.lora import LoRALinear as RefLoRA
from microlab.model.reference.lora import quantize_dequantize as ref_qdq


def test_lora_is_noop_at_init():
    torch.manual_seed(0)
    base = nn.Linear(16, 24)
    lora = LoRALinear(base, rank=4, alpha=8).eval()
    x = torch.randn(3, 16)
    assert torch.allclose(lora(x), base(x), atol=1e-6)  # B=0 -> identity


def test_lora_forward_matches_reference():
    torch.manual_seed(0)
    base = nn.Linear(16, 24)
    stu = LoRALinear(base, rank=4, alpha=8)
    ref = RefLoRA(base, rank=4, alpha=8)
    ref.load_state_dict(stu.state_dict())  # same weights
    stu.eval(), ref.eval()
    x = torch.randn(5, 16)
    stu.lora_B.data.normal_()
    ref.lora_B.data.copy_(stu.lora_B.data)
    assert torch.allclose(stu(x), ref(x), atol=1e-6)


def test_lora_merge_matches_forward():
    torch.manual_seed(0)
    base = nn.Linear(16, 24)
    lora = LoRALinear(base, rank=4, alpha=8)
    lora.lora_B.data.normal_()
    lora.eval()
    x = torch.randn(5, 16)
    w = lora.merged_weight()
    merged = nn.functional.linear(x, w, base.bias)
    assert torch.allclose(lora(x), merged, atol=1e-5)


def test_quantize_dequantize_matches_reference():
    torch.manual_seed(0)
    w = torch.randn(64, 32)
    for bits in [2, 4, 8]:
        assert torch.allclose(quantize_dequantize(w, bits), ref_qdq(w, bits), atol=1e-6)


def test_quantize_more_bits_lower_error():
    torch.manual_seed(0)
    w = torch.randn(64, 32)
    e4 = (quantize_dequantize(w, 4) - w).abs().mean()
    e8 = (quantize_dequantize(w, 8) - w).abs().mean()
    assert e8 < e4

pytestmark = pytest.mark.exercise
