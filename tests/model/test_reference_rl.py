import pytest
import torch

from microlab.model.reference.rl import (
    extract_answer,
    group_normalized_advantages,
    kl_to_reference,
    ppo_clip_loss,
    verifiable_reward,
)


def test_extract_answer_hash_form():
    assert extract_answer("... so the total is #### 42") == "42"
    assert extract_answer("step by step #### -17") == "-17"


def test_extract_answer_last_number_fallback():
    # No '####' marker -> take the last number in the text.
    assert extract_answer("first 3 then 7 and finally 11") == "11"


def test_extract_answer_commas_and_trailing_dot():
    assert extract_answer("the total is 1,234 dollars") == "1234"
    assert extract_answer("#### 1,000,000") == "1000000"


def test_extract_answer_none_when_no_numbers():
    assert extract_answer("no numbers here") is None


def test_verifiable_reward_match():
    assert verifiable_reward("the answer is #### 18", "#### 18") == 1.0
    # Match via last-number fallback on the generated side vs '####' gold.
    assert verifiable_reward("so it must be 18", "#### 18") == 1.0


def test_verifiable_reward_mismatch():
    assert verifiable_reward("#### 17", "#### 18") == 0.0


def test_verifiable_reward_unparseable():
    assert verifiable_reward("nope", "#### 18") == 0.0


def test_group_advantages_normalized():
    a = group_normalized_advantages(torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert a.mean().abs() < 1e-5
    assert abs(a.std(unbiased=True).item() - 1.0) < 0.2


def test_group_advantages_degenerate_no_nan():
    a = group_normalized_advantages(torch.tensor([1.0, 1.0, 1.0]))
    assert torch.isfinite(a).all()
    assert a.abs().max() < 1e-3  # all-equal rewards -> ~0 advantages


def test_ppo_clip_at_ratio_one():
    lp = torch.zeros(4)
    adv = torch.tensor([1.0, -1.0, 2.0, -2.0])
    # ratio == 1 everywhere -> loss is just -mean(advantages).
    assert ppo_clip_loss(lp, lp, adv).item() == pytest.approx(-adv.mean().item(), abs=1e-6)


def test_ppo_clips_large_positive_advantage():
    old = torch.zeros(1)
    new = torch.tensor([1.0])  # ratio = e >> 1.2
    adv = torch.tensor([1.0])
    # For A > 0 the clipped branch wins: loss == -(1+eps)*A == -1.2.
    assert ppo_clip_loss(new, old, adv, clip_eps=0.2).item() == pytest.approx(-1.2, abs=1e-5)


def test_ppo_large_negative_advantage_not_clipped_upward():
    # For A < 0 with ratio >> 1, min picks the (more negative) unclipped branch:
    # min(ratio*A, 1.2*A) = ratio*A since A<0 -> loss = -ratio*A = e * 1.
    old = torch.zeros(1)
    new = torch.tensor([1.0])
    adv = torch.tensor([-1.0])
    expected = -(torch.e * -1.0)  # = +e
    assert ppo_clip_loss(new, old, adv, clip_eps=0.2).item() == pytest.approx(expected, abs=1e-4)


def test_kl_to_reference_zero_when_equal():
    lp = torch.randn(6)
    assert kl_to_reference(lp, lp).item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.gpu
def test_policy_step_reduces_ppo_loss_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    torch.manual_seed(0)
    logits = torch.randn(8, 32, requires_grad=True, device="cuda")
    old = torch.log_softmax(logits.detach(), dim=-1)
    actions = torch.randint(0, 32, (8,), device="cuda")
    adv = torch.randn(8, device="cuda")
    opt = torch.optim.SGD([logits], lr=0.1)

    def loss_fn():
        lp = torch.log_softmax(logits, dim=-1)[torch.arange(8, device="cuda"), actions]
        olp = old[torch.arange(8, device="cuda"), actions]
        return ppo_clip_loss(lp, olp, adv)

    first = loss_fn().item()
    for _ in range(20):
        opt.zero_grad()
        loss_fn().backward()
        opt.step()
    assert loss_fn().item() <= first
    _ = kl_to_reference(torch.zeros(4, device="cuda"), torch.zeros(4, device="cuda"))
