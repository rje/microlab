import pytest
import torch

from microlab.data.reference.dataset import get_batch


def test_batch_shapes_and_shift():
    data = torch.arange(100)
    g = torch.Generator().manual_seed(0)
    x, y = get_batch(data, block_size=8, batch_size=4, generator=g)
    assert x.shape == (4, 8) and y.shape == (4, 8)
    # y is x shifted by one position
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_batch_is_deterministic_with_generator():
    data = torch.arange(100)
    x1, _ = get_batch(data, 8, 4, generator=torch.Generator().manual_seed(1))
    x2, _ = get_batch(data, 8, 4, generator=torch.Generator().manual_seed(1))
    assert torch.equal(x1, x2)


@pytest.mark.gpu
def test_batch_moves_to_cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    data = torch.arange(1000)
    x, y = get_batch(data, 16, 8, device="cuda")
    assert x.is_cuda and y.is_cuda and x.shape == (8, 16)
