import torch

from microlab.infer.reference.speculative import speculative_accept


def test_identical_distributions_accept_everything():
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(4, 16), dim=-1)
    tokens = probs.argmax(-1)
    n, fix = speculative_accept(tokens, probs, probs, torch.Generator().manual_seed(0))
    assert n == 4 and fix is None


def test_target_zero_prob_rejects_at_that_position():
    V = 8
    draft = torch.full((2, V), 1.0 / V)
    target = draft.clone()
    tokens = torch.tensor([3, 5])
    target[0, 3] = 0.0  # target hates the first draft token
    target[0] /= target[0].sum()
    n, fix = speculative_accept(tokens, draft, target, torch.Generator().manual_seed(0))
    assert n == 0 and fix is not None and fix.item() != 3


def test_resample_comes_from_positive_residual():
    draft = torch.tensor([[0.97, 0.01, 0.01, 0.01]])
    target = torch.tensor([[0.01, 0.49, 0.25, 0.25]])
    tokens = torch.tensor([0])
    g = torch.Generator().manual_seed(0)
    for _ in range(30):
        n, fix = speculative_accept(tokens, draft, target, g)
        if n == 0:
            assert fix.item() != 0  # residual max(0, p_t - p_d) is zero at token 0
