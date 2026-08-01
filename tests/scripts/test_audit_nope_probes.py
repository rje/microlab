"""scripts/audit_nope_probes.py pure logic: probe dataset assembly, sequence-level
train/test splitting, probe metrics + trainability on synthetic position-coded features,
feature collection via hooks on a tiny VariantGPT, and attention-row extraction (causal
masking, RoPE parity with the module forward) plus entropy/distance summaries. CPU only;
loaded via importlib since scripts/ isn't a package."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from microlab.model.reference.variants import VariantConfig, VariantGPT

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


anp = _load("audit_nope_probes")


def _tiny_model(pos: str) -> VariantGPT:
    torch.manual_seed(0)
    model = VariantGPT(VariantConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2,
                                     n_embd=16, dropout=0.0, norm="rms", pos=pos,
                                     mlp="swiglu"))
    return model.eval()


# ------------------------------------------------------------------ probe dataset logic

def test_features_to_dataset_flattens_and_labels_positions():
    feats = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)  # (N, T, C)
    x, y = anp.features_to_dataset(feats)
    assert x.shape == (6, 4) and y.shape == (6,)
    assert torch.equal(y, torch.tensor([0, 1, 2, 0, 1, 2]))
    assert torch.equal(x[0], feats[0, 0]) and torch.equal(x[5], feats[1, 2])


def test_split_sequences_is_disjoint_deterministic_and_seed_sensitive():
    tr, te = anp.split_sequences(10, n_test=3, seed=0)
    tr2, te2 = anp.split_sequences(10, n_test=3, seed=0)
    assert torch.equal(tr, tr2) and torch.equal(te, te2)
    assert len(te) == 3 and len(tr) == 7
    assert set(tr.tolist()) | set(te.tolist()) == set(range(10))
    assert set(tr.tolist()) & set(te.tolist()) == set()
    _, te3 = anp.split_sequences(10, n_test=3, seed=1)
    assert not torch.equal(te, te3)


def test_split_sequences_rejects_degenerate_split():
    with pytest.raises(ValueError):
        anp.split_sequences(4, n_test=4, seed=0)
    with pytest.raises(ValueError):
        anp.split_sequences(4, n_test=0, seed=0)


def test_probe_metrics_accuracy_and_mean_absolute_distance():
    preds = torch.tensor([0, 5, 2, 9])
    labels = torch.tensor([0, 3, 2, 1])
    m = anp.probe_metrics(preds, labels)
    assert m["acc"] == pytest.approx(0.5)
    assert m["mad"] == pytest.approx((0 + 2 + 0 + 8) / 4)


def test_train_position_probe_learns_position_coded_features():
    # Features literally one-hot encode the position (plus small noise): the probe must
    # solve it; the same probe on shuffled labels must stay near chance.
    T, C, n_seq = 8, 8, 24
    g = torch.Generator().manual_seed(0)
    feats = torch.eye(T).repeat(n_seq, 1, 1) + 0.01 * torch.randn(n_seq, T, C, generator=g)
    tr, te = anp.split_sequences(n_seq, n_test=8, seed=0)
    real = anp.train_position_probe(feats[tr], feats[te], n_positions=T, hidden=16,
                                    epochs=60, batch_size=64, lr=1e-2, seed=0,
                                    device="cpu")
    assert real["acc"] > 0.95
    assert real["mad"] < 0.1
    shuf = anp.train_position_probe(feats[tr], feats[te], n_positions=T, hidden=16,
                                    epochs=60, batch_size=64, lr=1e-2, seed=0,
                                    device="cpu", shuffle_labels=True)
    assert shuf["acc"] < 0.5
    assert shuf["mad"] > real["mad"]
    assert shuf["shuffled"] is True and real["shuffled"] is False


# ------------------------------------------------------------------- feature collection

def test_collect_features_taps_embedding_and_block_outputs():
    model = _tiny_model("nope")
    x = torch.randint(0, 64, (3, 16), generator=torch.Generator().manual_seed(1))
    feats = anp.collect_features(model, x, taps=["emb", 0, 1])
    assert set(feats) == {"emb", 0, 1}
    for v in feats.values():
        assert v.shape == (3, 16, 16) and v.dtype == torch.float32
    # "emb" is the post-dropout token embedding (dropout 0 -> exactly wte)
    assert torch.allclose(feats["emb"], model.transformer.wte(x).float(), atol=1e-6)
    # block taps are the residual-stream outputs of each block, pre-ln_f
    h = model.transformer.drop(model.transformer.wte(x))
    for i, block in enumerate(model.transformer.h):
        h = block(h)
        assert torch.allclose(feats[i], h.float(), atol=1e-5), f"tap {i}"


def test_collect_features_rejects_unknown_tap():
    model = _tiny_model("nope")
    x = torch.zeros(1, 8, dtype=torch.long)
    with pytest.raises(ValueError, match="tap"):
        anp.collect_features(model, x, taps=["final"])


# ------------------------------------------------------------------------ attention rows

@pytest.mark.parametrize("pos", ["nope", "rope"])
def test_attention_rows_are_causal_normalized_and_match_module_forward(pos):
    model = _tiny_model(pos)
    attn = model.transformer.h[0].attn
    g = torch.Generator().manual_seed(2)
    x_norm = torch.randn(2, 16, 16, generator=g)
    qpos = [0, 5, 15]
    rows = anp.attention_rows(attn, x_norm, qpos)
    assert rows.shape == (2, 2, 3, 16)  # (B, n_head, n_query, T)
    assert torch.allclose(rows.sum(-1), torch.ones(2, 2, 3), atol=1e-5)
    for qi, p in enumerate(qpos):
        assert rows[:, :, qi, p + 1:].abs().sum().item() == 0.0  # no future mass
    # Reconstruction: rows @ v pushed through c_proj must equal the module's own forward
    # at those query positions (proves q/k/v extraction + RoPE application are faithful).
    q, k, v = attn.c_attn(x_norm).split(attn.n_embd, dim=2)
    v = v.view(2, 16, attn.n_head, 8).transpose(1, 2)
    y = (rows @ v).transpose(1, 2).reshape(2, 3, 16)
    expected = attn(x_norm)[:, qpos, :]
    assert torch.allclose(attn.c_proj(y), expected, atol=1e-5)


def test_attention_rows_rejects_out_of_range_query():
    attn = _tiny_model("nope").transformer.h[0].attn
    with pytest.raises(ValueError, match="query"):
        anp.attention_rows(attn, torch.randn(1, 8, 16), [8])


# ------------------------------------------------------------------- entropy summaries

def test_row_entropy_uniform_and_delta():
    uniform = torch.full((1, 1, 1, 8), 1 / 8)
    assert anp.row_entropy(uniform).item() == pytest.approx(math.log(8), abs=1e-6)
    delta = torch.zeros(1, 1, 1, 8)
    delta[..., 3] = 1.0
    assert anp.row_entropy(delta).item() == pytest.approx(0.0, abs=1e-8)


def test_attention_summary_uniform_row_closed_form():
    # Query at absolute position 7 with uniform attention over its 8 visible keys.
    T, p = 12, 7
    rows = torch.zeros(1, 1, 1, T)
    rows[..., : p + 1] = 1 / (p + 1)
    s = anp.attention_summary(rows, [p], last_k=4)
    (entry,) = s
    assert entry["qpos"] == p
    assert entry["entropy"] == pytest.approx(math.log(p + 1), abs=1e-6)
    assert entry["entropy_norm"] == pytest.approx(1.0, abs=1e-6)
    # E[p - j] for j uniform on 0..p is p/2; mass on the last 4 keys is 4/8
    assert entry["mean_dist"] == pytest.approx(p / 2, abs=1e-5)
    assert entry["last_k_mass"] == pytest.approx(4 / 8, abs=1e-6)
    assert len(entry["entropy_norm_per_head"]) == 1


def test_attention_summary_delta_row_is_sharp():
    T, p = 10, 6
    rows = torch.zeros(2, 3, 1, T)  # batch 2, 3 heads
    rows[..., 0, p] = 1.0  # all mass on self
    (entry,) = anp.attention_summary(rows, [p], last_k=2)
    assert entry["entropy"] == pytest.approx(0.0, abs=1e-7)
    assert entry["entropy_norm"] == pytest.approx(0.0, abs=1e-7)
    assert entry["mean_dist"] == pytest.approx(0.0, abs=1e-7)
    assert entry["last_k_mass"] == pytest.approx(1.0, abs=1e-7)
