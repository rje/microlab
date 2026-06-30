import pytest
import torch

from microlab.model.reference.gpt import GPT, GPTConfig


def _tiny():
    return GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32))


def test_forward_shapes_and_loss():
    torch.manual_seed(0)
    m = _tiny()
    idx = torch.randint(0, 64, (4, 16))
    logits, loss = m(idx, idx)
    assert logits.shape == (4, 16, 64)
    assert loss.ndim == 0 and loss.item() > 0


def test_forward_without_targets_has_no_loss():
    m = _tiny()
    logits, loss = m(torch.randint(0, 64, (2, 8)))
    assert logits.shape == (2, 8, 64) and loss is None


def test_rejects_sequence_longer_than_block():
    m = _tiny()
    with pytest.raises(AssertionError):
        m(torch.randint(0, 64, (1, 17)))


def test_is_causal_changing_last_token_leaves_earlier_logits_unchanged():
    torch.manual_seed(0)
    m = _tiny()
    m.eval()
    idx = torch.randint(0, 64, (1, 16))
    a, _ = m(idx)
    idx2 = idx.clone()
    idx2[0, -1] = (idx2[0, -1] + 1) % 64
    b, _ = m(idx2)
    # earlier positions can't see the changed final token
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)


def test_weight_tying():
    m = _tiny()
    assert m.transformer.wte.weight is m.lm_head.weight


def test_deterministic_forward():
    torch.manual_seed(1)
    m = _tiny()
    m.eval()
    idx = torch.randint(0, 64, (2, 16))
    a, _ = m(idx)
    b, _ = m(idx)
    assert torch.allclose(a, b)
