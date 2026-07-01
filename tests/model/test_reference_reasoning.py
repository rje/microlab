import pytest
import torch

from microlab.model.reference.reasoning import (
    distillation_loss,
    filter_correct_traces,
    self_consistency,
)


def _last_int(s):
    import re

    m = re.findall(r"-?\d+", s)
    return m[-1] if m else None


def test_filter_correct_traces():
    traces = ["think... 42", "wrong... 41", "yes 42"]
    assert filter_correct_traces(traces, "42", _last_int) == ["think... 42", "yes 42"]


def test_self_consistency_majority():
    assert self_consistency(["42", "42", "41"]) == "42"
    assert self_consistency([]) is None


def test_distillation_zero_when_equal():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 16)
    assert distillation_loss(logits, logits).item() == pytest.approx(0.0, abs=1e-6)


def test_distillation_decreases_toward_teacher():
    torch.manual_seed(0)
    teacher = torch.randn(2, 5, 16)
    far = torch.randn(2, 5, 16)
    near = teacher + 0.01 * torch.randn(2, 5, 16)
    assert distillation_loss(near, teacher).item() < distillation_loss(far, teacher).item()


@pytest.mark.gpu
def test_distill_student_to_teacher_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from microlab.model.reference.gpt import GPT, GPTConfig

    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=64)
    teacher = GPT(cfg).cuda().eval()
    student = GPT(cfg).cuda()
    opt = torch.optim.AdamW(student.parameters(), lr=1e-3)
    x = torch.randint(0, 64, (4, 16), device="cuda")
    with torch.no_grad():
        tlog = teacher(x)[0]
    first = None
    for _ in range(60):
        loss = distillation_loss(student(x)[0], tlog)
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first
