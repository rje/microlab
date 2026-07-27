"""scripts/analyze_gqa_grouping.py: pure logic for the GQA-conversion audit — head
similarity matrices, exact partition search, Procrustes head alignment, and the
generalized (permuted / rotated / selected) MHA -> GQA conversion. Loaded via importlib
since scripts/ isn't a package."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from microlab.model.reference.variants import VariantConfig, VariantGPT, apply_rope

_here = Path(__file__).resolve().parents[2] / "scripts"
_SPEC = importlib.util.spec_from_file_location("analyze_gqa_script",
                                               _here / "analyze_gqa_grouping.py")
ag = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ag)


def _vcfg(n_kv_head=None, n_head=4, n_embd=32):
    return VariantConfig(vocab_size=64, block_size=32, n_layer=2, n_head=n_head,
                         n_embd=n_embd, norm="rms", pos="rope", mlp="swiglu",
                         n_kv_head=n_kv_head)


def _rand_orthogonal(d: int, seed: int) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=torch.Generator().manual_seed(seed)))
    return q


def _rand_plane_rotation(d: int, seed: int) -> torch.Tensor:
    """A random member of the RoPE-commuting family: independent 2-D rotations in the
    (i, i + d/2) planes — built by hand, independently of rope_plane_rotation."""
    half = d // 2
    phi = torch.rand(half, generator=torch.Generator().manual_seed(seed)) * 2 * math.pi
    rot = torch.zeros(d, d)
    idx = torch.arange(half)
    rot[idx, idx] = phi.cos()
    rot[idx + half, idx + half] = phi.cos()
    rot[idx, idx + half] = -phi.sin()
    rot[idx + half, idx] = phi.sin()
    return rot


# ---------------------------------------------------------------------------
# similarity + partition search
# ---------------------------------------------------------------------------

def test_head_cosine_matrix_orthogonal_and_identical():
    # heads 0,1 orthogonal; head 2 == head 0 -> cosine 1 with head 0, 0 with head 1
    w = torch.zeros(6, 2)  # 3 heads x head_dim 2, in_dim 2
    w[0, 0] = 1.0  # head 0
    w[3, 1] = 1.0  # head 1
    w[4, 0] = 1.0  # head 2 == head 0
    sim = ag.head_cosine_matrix(w, n_head=3)
    assert torch.allclose(sim.diagonal(), torch.ones(3))
    assert sim[0, 1] == pytest.approx(0.0)
    assert sim[0, 2] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="divisible"):
        ag.head_cosine_matrix(torch.zeros(5, 2), n_head=3)


def test_activation_cosine_matrix_hand_case():
    # two tokens; heads: a==b always, c orthogonal to both
    acts = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                         [[0.0, 2.0], [0.0, 2.0], [2.0, 0.0]]])
    sim = ag.activation_cosine_matrix(acts)
    assert sim[0, 1] == pytest.approx(1.0)
    assert sim[0, 2] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="expected"):
        ag.activation_cosine_matrix(torch.zeros(4, 2))


def test_group_score_and_medoid():
    sim = [[1.0, 0.9, 0.1, 0.0],
           [0.9, 1.0, 0.2, 0.0],
           [0.1, 0.2, 1.0, 0.8],
           [0.0, 0.0, 0.8, 1.0]]
    assert ag.group_score(sim, [[0, 1], [2, 3]]) == pytest.approx((0.9 + 0.8) / 2)
    assert ag.group_score(sim, [[0, 1, 2, 3]]) == pytest.approx(
        (0.9 + 0.1 + 0.0 + 0.2 + 0.0 + 0.8) / 6)
    with pytest.raises(ValueError, match="size 1"):
        ag.group_score(sim, [[0], [1], [2], [3]])
    # medoid: heads 1 and 2 tie on similarity mass (1.1); the tie keeps the lowest index
    assert ag.medoid(sim, [0, 1, 2, 3]) == 1
    assert ag.medoid(sim, [1, 2, 3]) == 2  # untied: 0.2+0.8 beats 0.2+0.0 and 0.0+0.8
    assert ag.medoid(sim, [3]) == 3


def test_best_partition_recovers_planted_clusters():
    # planted pairs (0,2) and (1,3) dominate every alternative pairing
    sim = [[1.0, 0.1, 0.9, 0.1],
           [0.1, 1.0, 0.1, 0.8],
           [0.9, 0.1, 1.0, 0.1],
           [0.1, 0.8, 0.1, 1.0]]
    groups, score = ag.best_partition(sim, 2)
    assert sorted(map(sorted, groups)) == [[0, 2], [1, 3]]
    assert score == pytest.approx((0.9 + 0.8) / 2)
    # group_size == n: the single partition, scored over all pairs
    groups, score = ag.best_partition(sim, 4)
    assert groups == [[0, 1, 2, 3]]
    assert score == pytest.approx(ag.group_score(sim, groups))
    with pytest.raises(ValueError, match=">= 2"):
        ag.best_partition(sim, 1)
    with pytest.raises(ValueError, match="divide"):
        ag.best_partition(sim, 3)


def test_best_partition_beats_adjacent_by_construction():
    # 6 heads, planted clusters {0,3,4} and {1,2,5}: exact search must find them and
    # score at least the adjacent grouping (it is optimal over ALL partitions)
    hi, lo = 0.9, 0.05
    clusters = [{0, 3, 4}, {1, 2, 5}]
    sim = [[hi if any(i in c and j in c for c in clusters) else lo for j in range(6)]
           for i in range(6)]
    groups, score = ag.best_partition(sim, 3)
    assert sorted(map(sorted, groups)) == [[0, 3, 4], [1, 2, 5]]
    assert score >= ag.group_score(sim, ag.adjacent_groups(6, 2))
    assert score == pytest.approx(hi)


def test_adjacent_groups():
    assert ag.adjacent_groups(14, 2) == [list(range(7)), list(range(7, 14))]
    assert ag.adjacent_groups(4, 4) == [[0], [1], [2], [3]]
    with pytest.raises(ValueError, match="divide"):
        ag.adjacent_groups(14, 4)


# ---------------------------------------------------------------------------
# Procrustes alignment
# ---------------------------------------------------------------------------

def test_procrustes_rotation_recovers_planted_orthogonal():
    d, n = 8, 200
    src = torch.randn(n, d, generator=torch.Generator().manual_seed(0))
    r = _rand_orthogonal(d, seed=1)
    tgt = src @ r.T  # tgt_n = R src_n
    q = ag.procrustes_rotation(src, tgt)
    assert torch.allclose(q, r, atol=1e-5)
    assert torch.allclose(q @ q.T, torch.eye(d), atol=1e-5)


def test_rope_plane_rotation_recovers_planted_and_commutes_with_rope():
    d, n = 8, 200
    src = torch.randn(n, d, generator=torch.Generator().manual_seed(2))
    r = _rand_plane_rotation(d, seed=3)
    q = ag.rope_plane_rotation(src, src @ r.T)
    assert torch.allclose(q, r, atol=1e-5)
    # the family commutes with RoPE: rotate-then-rope == rope-then-rotate
    from microlab.model.reference.variants import build_rope_cache

    cos, sin = build_rope_cache(16, d)
    x = torch.randn(1, 1, 16, d, generator=torch.Generator().manual_seed(4))
    a = apply_rope(x @ q.T, cos, sin)
    b = apply_rope(x, cos, sin) @ q.T
    assert torch.allclose(a, b, atol=1e-5)
    # a generic orthogonal does NOT commute (the constraint is load-bearing)
    g = _rand_orthogonal(d, seed=5)
    assert not torch.allclose(apply_rope(x @ g.T, cos, sin),
                              apply_rope(x, cos, sin) @ g.T, atol=1e-3)


def test_aligned_similarity_sees_through_rotation():
    # head 1 is an exact rotation of head 0: raw cosine is far from 1, aligned cosine
    # is 1 (full orthogonal family) — the gauge-invariance point of arXiv 2412.20677
    d, n = 8, 300
    base = torch.randn(n, d, generator=torch.Generator().manual_seed(6))
    r = _rand_plane_rotation(d, seed=7)
    acts = torch.stack([base, base @ r.T], dim=1)
    raw = ag.activation_cosine_matrix(acts)[0, 1]
    aligned = ag.aligned_similarity_matrix(acts, constrained=False)[0, 1]
    assert aligned == pytest.approx(1.0, abs=1e-4)
    assert aligned > raw + 0.1
    # constrained variant also recovers it (the rotation is in the RoPE family)
    aligned_c = ag.aligned_similarity_matrix(acts, constrained=True)[0, 1]
    assert aligned_c == pytest.approx(1.0, abs=1e-4)


def test_align_group_rotations_map_members_to_common_frame():
    d, n = 8, 300
    base = torch.randn(n, d, generator=torch.Generator().manual_seed(8))
    r1 = _rand_plane_rotation(d, seed=9)
    acts = torch.stack([base, base @ r1.T], dim=1)
    rots = ag.align_group(acts, [0, 1], constrained=True)
    aligned0 = acts[:, 0] @ rots[0].T
    aligned1 = acts[:, 1] @ rots[1].T
    assert torch.allclose(aligned0, aligned1, atol=1e-4)


# ---------------------------------------------------------------------------
# generalized conversion
# ---------------------------------------------------------------------------

def test_adjacent_mean_matches_convert_state_dict():
    torch.manual_seed(0)
    sd = VariantGPT(_vcfg()).state_dict()
    a = ag.convert_with_groups(sd, n_head=4, n_embd=32, groups=ag.adjacent_groups(4, 2))
    b = ag.convert_gqa.convert_state_dict(sd, n_head=4, n_embd=32, n_kv_head=2)
    assert set(a) == set(b)
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-7), k


def test_singleton_groups_any_order_preserve_logits():
    # scrambled singleton groups = a pure head permutation (q rows, kv blocks, c_proj
    # columns): attention is equivariant to it, so logits must match MHA exactly
    torch.manual_seed(1)
    mha = VariantGPT(_vcfg()).eval()
    new = ag.convert_with_groups(mha.state_dict(), n_head=4, n_embd=32,
                                 groups=[[2], [0], [3], [1]])
    gqa = VariantGPT(_vcfg(n_kv_head=4)).eval()
    gqa.load_state_dict(new)
    x = torch.randint(0, 64, (2, 16), generator=torch.Generator().manual_seed(2))
    la, _ = mha(x)
    lb, _ = gqa(x)
    assert torch.allclose(la, lb, atol=1e-4)


def test_function_preserving_rotations_preserve_logits():
    # per-head rotations (RoPE-commuting on K/Q, arbitrary orthogonal on V absorbed into
    # c_proj) with singleton groups must leave the model function unchanged — validates
    # rotate_heads, absorb_output_rotation, and the RoPE-commutation constraint together
    torch.manual_seed(3)
    n_head, n_embd = 4, 32
    d = n_embd // n_head
    mha = VariantGPT(_vcfg()).eval()
    rots = {li: {"k": torch.stack([_rand_plane_rotation(d, seed=10 * li + h)
                                   for h in range(n_head)]),
                 "v": torch.stack([_rand_orthogonal(d, seed=100 + 10 * li + h)
                                   for h in range(n_head)])}
            for li in range(2)}
    new = ag.convert_with_groups(mha.state_dict(), n_head=n_head, n_embd=n_embd,
                                 groups=[[0], [1], [2], [3]], rotations=rots)
    gqa = VariantGPT(_vcfg(n_kv_head=4)).eval()
    gqa.load_state_dict(new)
    x = torch.randint(0, 64, (2, 16), generator=torch.Generator().manual_seed(4))
    la, _ = mha(x)
    lb, _ = gqa(x)
    assert torch.allclose(la, lb, atol=1e-4), f"max diff {(la - lb).abs().max()}"


def test_scattered_identical_heads_pool_losslessly():
    # make head 3 := head 0 and head 2 := head 1 (K and V), then pool with the matching
    # NON-adjacent partition [[0,3],[1,2]] — conversion must be exact
    torch.manual_seed(5)
    n_head, n_embd = 4, 32
    d = n_embd // n_head
    mha = VariantGPT(_vcfg()).eval()
    with torch.no_grad():
        for block in mha.transformer.h:
            w, b = block.attn.c_attn.weight, block.attn.c_attn.bias
            for base in (n_embd, 2 * n_embd):  # K rows, then V rows
                hw = w[base:base + n_embd].view(n_head, d, n_embd)
                hb = b[base:base + n_embd].view(n_head, d)
                hw[3], hb[3] = hw[0].clone(), hb[0].clone()
                hw[2], hb[2] = hw[1].clone(), hb[1].clone()
    new = ag.convert_with_groups(mha.state_dict(), n_head=n_head, n_embd=n_embd,
                                 groups=[[0, 3], [1, 2]])
    gqa = VariantGPT(_vcfg(n_kv_head=2)).eval()
    gqa.load_state_dict(new)
    x = torch.randint(0, 64, (2, 16), generator=torch.Generator().manual_seed(6))
    la, _ = mha(x)
    lb, _ = gqa(x)
    assert torch.allclose(la, lb, atol=1e-4), f"max diff {(la - lb).abs().max()}"


def test_select_mode_keeps_exact_head_blocks():
    torch.manual_seed(7)
    n_head, n_embd = 4, 32
    d = n_embd // n_head
    sd = VariantGPT(_vcfg()).state_dict()
    new = ag.convert_with_groups(sd, n_head=n_head, n_embd=n_embd,
                                 groups=[[0, 1], [2, 3]], mode="select", select=[1, 2])
    kw = sd["transformer.h.0.attn.c_attn.weight"][n_embd:2 * n_embd].view(n_head, d, n_embd)
    got = new["transformer.h.0.attn.kv_proj.weight"][:2 * d].view(2, d, n_embd)
    assert torch.equal(got[0], kw[1])
    assert torch.equal(got[1], kw[2])
    with pytest.raises(ValueError, match="not a member"):
        ag.convert_with_groups(sd, n_head=n_head, n_embd=n_embd, groups=[[0, 1], [2, 3]],
                               mode="select", select=[2, 3])


def test_renorm_restores_mean_member_norm():
    torch.manual_seed(8)
    sd = VariantGPT(_vcfg()).state_dict()
    n_head, n_embd = 4, 32
    d = n_embd // n_head
    new = ag.convert_with_groups(sd, n_head=n_head, n_embd=n_embd,
                                 groups=[[0, 1], [2, 3]], renorm=True)
    kw = sd["transformer.h.0.attn.c_attn.weight"][n_embd:2 * n_embd].view(n_head, d, n_embd)
    got = new["transformer.h.0.attn.kv_proj.weight"][:2 * d].view(2, d, n_embd)
    for gi, g in enumerate([[0, 1], [2, 3]]):
        member_norm = torch.stack([kw[h].norm() for h in g]).mean()
        assert float(got[gi].norm()) == pytest.approx(float(member_norm), rel=1e-5)


def test_convert_with_groups_validation():
    sd = VariantGPT(_vcfg()).state_dict()
    kw = dict(n_head=4, n_embd=32)
    with pytest.raises(ValueError, match="equal-size"):
        ag.convert_with_groups(sd, **kw, groups=[[0], [1, 2, 3]])
    with pytest.raises(ValueError, match="cover"):
        ag.convert_with_groups(sd, **kw, groups=[[0, 1], [2, 2]])
    with pytest.raises(ValueError, match="requires"):
        ag.convert_with_groups(sd, **kw, groups=[[0, 1], [2, 3]], mode="select")
    with pytest.raises(ValueError, match="only meaningful"):
        ag.convert_with_groups(sd, **kw, groups=[[0, 1], [2, 3]], select=[0, 2])
    with pytest.raises(ValueError, match="mutually exclusive"):
        ag.convert_with_groups(sd, **kw, groups=[[0, 1], [2, 3]],
                               scale_correct=True, renorm=True)
    with pytest.raises(ValueError, match="unknown mode"):
        ag.convert_with_groups(sd, **kw, groups=[[0, 1], [2, 3]], mode="median")
    with pytest.raises(ValueError, match="not an MHA"):
        ag.convert_with_groups({"transformer.wte.weight": torch.zeros(4, 4)}, **kw,
                               groups=[[0, 1], [2, 3]])
