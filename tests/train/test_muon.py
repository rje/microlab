"""Muon optimizer (Newton-Schulz orthogonalized momentum) + hybrid Muon/AdamW grouping.

Muon orthogonalizes the SGD-momentum update of each 2-D matrix via a quintic
Newton-Schulz iteration (Jordan et al.; arXiv 2502.16982). Non-matrix params
(embeddings, the tied wte/lm_head tensor, norms) stay on AdamW.
"""

import pytest
import torch

from microlab.model.reference.variants import VariantConfig, VariantGPT
from microlab.train.config import RunConfig
from microlab.train.muon import Muon, build_muon_param_groups, newton_schulz
from microlab.train.trainer import TensorData, Trainer


def _cfg(**kw):
    base = dict(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32,
                warmup_steps=5, max_steps=20, lr_decay_steps=20, batch_size=8,
                eval_interval=1000, ckpt_interval=1000, device="cpu", dtype="float32")
    base.update(kw)
    return RunConfig(**base)


def _small_model():
    return VariantGPT(VariantConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2,
                                    n_embd=32, norm="rms", pos="rope", mlp="swiglu"))


def _gram_dev(u: torch.Tensor) -> float:
    """Max abs deviation of the small-side gram matrix from identity. For a tall matrix
    the orthogonalized result is semi-orthogonal: U.T @ U ~ I over the column space (the
    row-space gram of a tall matrix is a rank-deficient projector, never near I)."""
    gram = u.T @ u if u.size(0) >= u.size(1) else u @ u.T
    n = gram.size(0)
    return (gram.float() - torch.eye(n)).abs().max().item()


def test_newton_schulz_near_orthogonal_tall_and_wide():
    # The quintic iteration with coefficients (3.4445, -4.7750, 2.0315) is approximate by
    # design: 5 steps drive singular values into roughly [0.68, 1.13] (the reference
    # impl's documented band; measured [0.68, 1.04] in fp32 across seeds/shapes), not
    # exactly to 1 — so the gram can deviate from I by up to ~1 - 0.68^2 ~= 0.54.
    g = torch.Generator().manual_seed(0)
    tall = torch.randn(64, 32, generator=g)
    wide = torch.randn(32, 64, generator=g)
    for m in (tall, wide):
        u = newton_schulz(m)
        assert u.shape == m.shape
        assert _gram_dev(u) < 0.55
    # sanity: the raw random matrix is nowhere near orthogonal
    assert _gram_dev(tall) > 1.0


def test_newton_schulz_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        newton_schulz(torch.randn(4, 4, 4))


def test_muon_rejects_non_2d_params():
    with pytest.raises(ValueError, match="2-D"):
        Muon([torch.nn.Parameter(torch.randn(8))])


def test_muon_orthogonalizes_the_update():
    # One Muon step from zero momentum applies -lr * scale * NS(grad): the applied update
    # must be (semi-)orthogonal even though the gradient is skewed. Skew capped at 10x —
    # 5 NS steps lift a singular value by at most ~3.44^5, so far-tinier directions
    # (1e-3+) legitimately stay unconverged in the reference-quality iteration.
    p = torch.nn.Parameter(torch.zeros(32, 16))
    opt = Muon([p], lr=1.0, momentum=0.0, nesterov=False, weight_decay=0.0)
    g = torch.randn(32, 16, generator=torch.Generator().manual_seed(1))
    p.grad = g * torch.logspace(0, -1, 16)  # skewed singular spectrum
    opt.step()
    scale = max(1.0, 32 / 16) ** 0.5
    assert _gram_dev(-p.detach() / scale) < 0.55


def test_tied_weight_in_exactly_one_group():
    # transformer.wte.weight IS lm_head.weight (one tensor). It must appear exactly once
    # across all groups — and in the AdamW group (Muon on the embedding is off-menu).
    model = _small_model()
    tied = model.lm_head.weight
    assert tied is model.transformer.wte.weight  # precondition: weights are tied
    muon_params, adamw_params = build_muon_param_groups(model)
    occurrences = sum(1 for p in [*muon_params, *adamw_params] if p is tied)
    assert occurrences == 1
    assert any(p is tied for p in adamw_params)


def test_param_groups_partition_all_params():
    # Every trainable param lands in exactly one group; Muon gets only 2-D block matrices.
    model = _small_model()
    muon_params, adamw_params = build_muon_param_groups(model)
    all_ids = {id(p) for p in model.parameters()}
    group_ids = [id(p) for p in [*muon_params, *adamw_params]]
    assert len(group_ids) == len(set(group_ids))  # disjoint
    assert set(group_ids) == all_ids              # complete
    assert muon_params, "expected transformer-block matrices in the Muon group"
    assert all(p.ndim == 2 for p in muon_params)
    block_ids = {id(p) for p in model.transformer.h.parameters()}
    assert all(id(p) in block_ids for p in muon_params)
    assert all(p.ndim != 2 or id(p) not in block_ids for p in adamw_params)


def test_unknown_optimizer_raises():
    data = TensorData(torch.randint(0, 64, (4000,)))
    with pytest.raises(ValueError, match="optimizer"):
        Trainer(_cfg(optimizer="sgd"), data, data)


def test_muon_training_reduces_loss_cpu():
    torch.manual_seed(0)
    data = TensorData(torch.randint(0, 64, (4000,)))
    tr = Trainer(_cfg(optimizer="muon"), data, data)
    stats = tr.train()
    assert stats["history"][-1] < stats["history"][0]


def test_muon_resume_equivalence_cpu(tmp_path):
    # An uninterrupted 20-step muon run must match a 10-step + resume + 10-step run
    # (model AND both optimizers' state round-trip through the checkpoint).
    data = TensorData(torch.randint(0, 64, (4000,), generator=torch.Generator().manual_seed(42)))
    a = Trainer(_cfg(optimizer="muon", max_steps=20), data, None)
    a.train()
    ck = str(tmp_path / "ck.pt")
    b = Trainer(_cfg(optimizer="muon", max_steps=10), data, None)
    b.train()
    b.save_checkpoint(ck)
    c = Trainer(_cfg(optimizer="muon", max_steps=20), data, None)
    c.load_checkpoint(ck)
    c.train()  # resumes from step 10 to 20
    for pa, pc in zip(a.model.parameters(), c.model.parameters(), strict=True):
        assert torch.allclose(pa, pc, atol=1e-5)


def test_muon_lr_scale_applied_to_matrix_group_only(tmp_path):
    # The schedule drives cfg.lr; Muon matrix groups run at muon_lr, i.e. scaled by
    # muon_lr / lr, while the aux-AdamW groups track cfg.lr exactly.
    data = TensorData(torch.randint(0, 64, (4000,)))
    tr = Trainer(_cfg(optimizer="muon", lr=1e-3, muon_lr=2e-2, warmup_steps=0), data, None)
    tr.train_step()
    lrs = sorted({g["lr"] for g in tr.optimizer.param_groups})
    from microlab.train.trainer import get_lr
    sched = get_lr(0, tr.cfg)
    assert lrs == sorted({sched, sched * 20.0})
