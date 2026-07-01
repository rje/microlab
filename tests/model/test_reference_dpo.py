import math

import pytest
import torch

from microlab.model.reference.dpo import IGNORE_INDEX, dpo_loss, sequence_logprob


def test_sequence_logprob_sums_masked_token_logprobs():
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 8)
    labels = torch.tensor([[IGNORE_INDEX, 2, 5, 1]])
    # manual: shift -> logits[:, :3] predict labels[:, 1:] = [2,5,1], all supervised
    lp = torch.log_softmax(logits[:, :-1], dim=-1)
    expected = lp[0, 0, 2] + lp[0, 1, 5] + lp[0, 2, 1]
    assert sequence_logprob(logits, labels).item() == pytest.approx(expected.item(), abs=1e-5)


def test_dpo_loss_zero_advantage_is_log2():
    z = torch.zeros(4)
    loss, _ = dpo_loss(z, z, z, z, beta=0.1)  # policy == ref -> logits 0 -> -log sigmoid 0
    assert loss.item() == pytest.approx(math.log(2), abs=1e-5)


def test_dpo_loss_prefers_chosen():
    # policy raises chosen and lowers rejected vs ref -> loss < log2, acc = 1
    pc, pr = torch.tensor([1.0]), torch.tensor([-1.0])
    rc, rr = torch.tensor([0.0]), torch.tensor([0.0])
    loss, acc = dpo_loss(pc, pr, rc, rr, beta=0.5)
    assert loss.item() < math.log(2) and acc == pytest.approx(1.0)


@pytest.mark.gpu
def test_dpo_step_reduces_loss_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    import copy

    from microlab.model.reference.gpt import GPT, GPTConfig
    torch.manual_seed(0)
    policy = GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=64)).cuda()
    ref = copy.deepcopy(policy).eval()
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    chosen = torch.randint(0, 64, (4, 16), device="cuda")
    rejected = torch.randint(0, 64, (4, 16), device="cuda")
    first = None
    for _ in range(60):
        pc = sequence_logprob(policy(chosen)[0], chosen)
        pr = sequence_logprob(policy(rejected)[0], rejected)
        with torch.no_grad():
            rc = sequence_logprob(ref(chosen)[0], chosen)
            rr = sequence_logprob(ref(rejected)[0], rejected)
        loss, _ = dpo_loss(pc, pr, rc, rr, beta=0.1)
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first
