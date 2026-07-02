import torch

from microlab.infer.reference.sampling import sample_next


def test_temperature_zero_is_argmax():
    logits = torch.tensor([[0.1, 2.0, -1.0], [3.0, 0.0, 0.0]])
    assert sample_next(logits, temperature=0.0).squeeze(1).tolist() == [1, 0]


def test_top_k_restricts_support():
    torch.manual_seed(0)
    logits = torch.tensor([[5.0, 4.0, -10.0, -10.0]])
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        tok = sample_next(logits, top_k=2, generator=g).item()
        assert tok in (0, 1)


def test_top_p_keeps_minimal_prefix():
    # probs ~ [0.7, 0.2, 0.06, 0.04]; top_p=0.8 keeps exactly {0, 1}
    probs = torch.tensor([[0.7, 0.2, 0.06, 0.04]])
    logits = probs.log()
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        assert sample_next(logits, top_p=0.8, generator=g).item() in (0, 1)


def test_generator_reproducible():
    logits = torch.randn(2, 16)
    a = sample_next(logits, generator=torch.Generator().manual_seed(7))
    b = sample_next(logits, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)
