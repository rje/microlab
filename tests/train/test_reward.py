"""microlab.train.reward: the Bradley-Terry loss math, the RewardModel scoring at the LAST
NON-PAD token of a padded batch (padding-invariant, last-token-sensitive), the reward collator,
and the reward-checkpoint save/load roundtrip (cfg + state_dict, servable-pattern)."""

from __future__ import annotations

import math

import pytest
import torch

from microlab.model.reference.variants import VariantConfig, VariantGPT
from microlab.train.reward import (
    RewardModel,
    bradley_terry_loss,
    collate_reward,
    load_reward_checkpoint,
    pairwise_accuracy,
    save_reward_checkpoint,
)


def _tiny_backbone(vocab: int = 64, block: int = 32) -> VariantGPT:
    torch.manual_seed(0)
    cfg = VariantConfig(vocab_size=vocab, block_size=block, n_layer=2, n_head=2, n_embd=16,
                        dropout=0.0, norm="rms", pos="rope", mlp="swiglu")
    return VariantGPT(cfg)


# ---------------------------------------------------------------- bradley_terry_loss


def test_bradley_terry_loss_known_values():
    z = torch.zeros(3)
    # Equal rewards: -log sigmoid(0) = log 2.
    assert abs(bradley_terry_loss(z, z).item() - math.log(2.0)) < 1e-6
    # Margin 1: -log sigmoid(1), exactly.
    got = bradley_terry_loss(torch.tensor([1.0]), torch.tensor([0.0])).item()
    want = -math.log(1.0 / (1.0 + math.exp(-1.0)))
    assert abs(got - want) < 1e-6
    # Correct ordering with a huge margin -> ~0 loss; inverted ordering -> ~margin.
    big = torch.full((3,), 20.0)
    assert bradley_terry_loss(big, z).item() < 1e-6
    assert bradley_terry_loss(z, big).item() > 19.0


def test_bradley_terry_loss_monotone_and_swap_symmetric():
    z = torch.zeros(1)
    l1 = bradley_terry_loss(torch.tensor([1.0]), z)
    l2 = bradley_terry_loss(torch.tensor([2.0]), z)
    assert l2 < l1 < bradley_terry_loss(z, z)  # larger margin -> smaller loss
    # -logsigmoid(m) + -logsigmoid(-m) == m + 2*(-logsigmoid(m))... simpler identity:
    # sigmoid(m) + sigmoid(-m) == 1, so exp(-loss(m)) + exp(-loss(-m)) == 1.
    m = torch.tensor([1.3])
    fwd = bradley_terry_loss(m, z).item()
    rev = bradley_terry_loss(z, m).item()
    assert abs(math.exp(-fwd) + math.exp(-rev) - 1.0) < 1e-6


def test_bradley_terry_loss_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        bradley_terry_loss(torch.zeros(2), torch.zeros(3))


def test_pairwise_accuracy_counts_strict_wins():
    r_c = torch.tensor([1.0, 0.0, 2.0, 1.0])
    r_r = torch.tensor([0.0, 1.0, 1.0, 1.0])  # win, loss, win, tie
    assert pairwise_accuracy(r_c, r_r) == 0.5  # ties are NOT wins


# ---------------------------------------------------------------- collate_reward


def test_collate_reward_pads_right_and_records_lengths():
    batch = collate_reward([[5, 6, 7], [8, 9]], pad_id=0)
    assert batch["input_ids"].tolist() == [[5, 6, 7], [8, 9, 0]]
    assert batch["lengths"].tolist() == [3, 2]
    assert batch["input_ids"].dtype == torch.long
    assert batch["lengths"].dtype == torch.long


def test_collate_reward_rejects_empty_sequence():
    with pytest.raises(ValueError, match="empty"):
        collate_reward([[1, 2], []], pad_id=0)


# ---------------------------------------------------------------- RewardModel


def test_reward_model_head_is_scalar_and_small_init():
    model = RewardModel(_tiny_backbone())
    assert model.head.weight.shape == (1, 16)
    assert model.head.bias is None
    assert model.head.weight.abs().max().item() < 0.2  # fresh small init, not backbone-scale


def test_reward_model_scores_last_non_pad_token():
    model = RewardModel(_tiny_backbone()).eval()
    seq_short, seq_long = [5, 6, 7], [5, 6, 7, 8, 9]
    batch = collate_reward([seq_short, seq_long], pad_id=0)
    with torch.no_grad():
        scores = model(batch["input_ids"], batch["lengths"])
        unpadded = model(torch.tensor([seq_short]), torch.tensor([3]))
    assert scores.shape == (2,)
    # Right-padding must not change the short sequence's score (causal attention + the score
    # being read at the last REAL token, not the last position).
    assert torch.allclose(scores[0], unpadded[0], atol=1e-5)


def test_reward_model_score_depends_on_last_real_token():
    # Two padded rows identical except at their last real token -> different scores; and a
    # rigged identity-like check: score must equal head(hidden at position length-1).
    model = RewardModel(_tiny_backbone()).eval()
    a = collate_reward([[5, 6, 7, 0, 0][:3], [9, 9, 9, 9, 9]], pad_id=0)  # last real = 7
    b = collate_reward([[5, 6, 8], [9, 9, 9, 9, 9]], pad_id=0)            # last real = 8
    with torch.no_grad():
        sa = model(a["input_ids"], a["lengths"])
        sb = model(b["input_ids"], b["lengths"])
    assert not torch.allclose(sa[0], sb[0])
    assert torch.allclose(sa[1], sb[1])  # the untouched row is untouched


def test_reward_model_matches_rigged_trunk_forward():
    # With the head rigged to sum the hidden state, the score must equal the summed ln_f
    # output at the last non-pad position of the backbone trunk — proving both the position
    # selection and that the score comes from the trunk's hidden state (not the LM head).
    model = RewardModel(_tiny_backbone()).eval()
    with torch.no_grad():
        model.head.weight.fill_(1.0)
    ids = torch.tensor([[3, 4, 5, 0, 0], [6, 7, 8, 9, 10]])
    lengths = torch.tensor([3, 5])
    t = model.backbone.transformer
    with torch.no_grad():
        x = t.drop(t.wte(ids))
        for block in t.h:
            x = block(x)
        x = t.ln_f(x)
        want = torch.stack([x[0, 2].sum(), x[1, 4].sum()])
        got = model(ids, lengths)
    assert torch.allclose(got, want, atol=1e-5)


def test_reward_model_validates_lengths_and_block():
    model = RewardModel(_tiny_backbone(block=8)).eval()
    ids = torch.tensor([[1, 2, 3]])
    with pytest.raises(ValueError, match="lengths"):
        model(ids, torch.tensor([0]))  # too small
    with pytest.raises(ValueError, match="lengths"):
        model(ids, torch.tensor([4]))  # beyond T
    with pytest.raises(ValueError, match="lengths"):
        model(ids, torch.tensor([3, 3]))  # batch mismatch
    with pytest.raises(ValueError, match="block_size"):
        model(torch.ones(1, 9, dtype=torch.long), torch.tensor([9]))


def test_reward_model_trains_end_to_end_gradients_flow():
    model = RewardModel(_tiny_backbone())
    batch_c = collate_reward([[1, 2, 3], [4, 5]], pad_id=0)
    batch_r = collate_reward([[6, 7], [8, 9, 10]], pad_id=0)
    loss = bradley_terry_loss(
        model(batch_c["input_ids"], batch_c["lengths"]),
        model(batch_r["input_ids"], batch_r["lengths"]),
    )
    loss.backward()
    assert model.head.weight.grad is not None
    assert model.backbone.transformer.wte.weight.grad is not None


# ---------------------------------------------------------------- checkpoint save/load


def test_reward_checkpoint_roundtrip(tmp_path):
    model = RewardModel(_tiny_backbone()).eval()
    path = tmp_path / "ckpt_7.pt"
    save_reward_checkpoint(path, model, step=7)
    loaded, step = load_reward_checkpoint(path, device="cpu")
    assert step == 7
    assert loaded.backbone.config.n_embd == 16
    assert loaded.backbone.config.norm == "rms"
    batch = collate_reward([[1, 2, 3], [4, 5]], pad_id=0)
    with torch.no_grad():
        assert torch.allclose(
            model(batch["input_ids"], batch["lengths"]),
            loaded(batch["input_ids"], batch["lengths"]),
        )


def test_reward_checkpoint_roundtrip_hybrid_backbone(tmp_path):
    # coder-1b's field profile scaled down (KDA:MLA hybrid, peri-LN, NoPE, MLA latents,
    # qk-norm). The hand-listed config rebuild this guards against dropped every one of
    # these fields, so a reward model saved on a hybrid backbone could not be reloaded.
    cfg = VariantConfig(vocab_size=64, block_size=32, n_layer=4, n_head=2, n_embd=16,
                        norm="rms", pos="nope", mlp="swiglu", block_norm="peri",
                        hybrid_every=4, gdn_gate="channel", global_attn="mla",
                        mla_kv_lora=8, qk_norm=True, gdn_fused=False)
    torch.manual_seed(0)
    model = RewardModel(VariantGPT(cfg)).eval()
    path = tmp_path / "ckpt_3.pt"
    save_reward_checkpoint(path, model, step=3)
    loaded, step = load_reward_checkpoint(path, device="cpu")  # strict load must pass
    got = loaded.backbone.config
    assert step == 3
    assert (got.hybrid_every, got.global_attn, got.block_norm) == (4, "mla", "peri")
    assert (got.mla_kv_lora, got.qk_norm, got.gdn_gate) == (8, True, "channel")
    batch = collate_reward([[1, 2, 3], [4, 5]], pad_id=0)
    with torch.no_grad():
        assert torch.allclose(
            model(batch["input_ids"], batch["lengths"]),
            loaded(batch["input_ids"], batch["lengths"]),
        )


def test_load_reward_checkpoint_rejects_non_reward_ckpt(tmp_path):
    # An LM checkpoint (no kind marker) must be refused loudly, not half-loaded.
    path = tmp_path / "ckpt_1.pt"
    torch.save({"model": {}, "cfg": None, "step": 1}, path)
    with pytest.raises(ValueError, match="reward"):
        load_reward_checkpoint(path, device="cpu")
