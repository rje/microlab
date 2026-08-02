"""SliceReader must read EVERY document in EVERY shard, and resume exactly.

The bug this guards: `doc_i` was never reset when a new shard loaded, so the
`doc_i >= len(docs)` test fired again immediately and dropped the shard whole. A
three-shard slice yielded shards 0 and 2 — a third of the corpus silently missing. It
surfaced only as slices "exhausting" at a fraction of their token count during the 21B
build, which is a symptom that is easy to explain away.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "build_mix", Path(__file__).resolve().parents[2] / "scripts" / "build_mix.py")
build_mix = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_mix)

EOT = 0


def _slice(tmp_path, docs_per_shard):
    """Write shards whose documents are uniquely identifiable by their first token."""
    import json
    shards, tag = [], 1
    for si, n in enumerate(docs_per_shard):
        toks = []
        for _ in range(n):
            toks += [tag, tag, EOT]
            tag += 1
        name = f"train-{si:05d}.bin"
        np.asarray(toks, dtype=np.uint16).tofile(tmp_path / name)
        shards.append({"file": name, "tokens": len(toks)})
    (tmp_path / "train-manifest.json").write_text(json.dumps(
        {"split": "train", "dtype": "uint16", "shards": shards,
         "total_tokens": sum(s["tokens"] for s in shards)}))
    return build_mix.SliceReader("t", tmp_path, "train", EOT)


def _drain(r):
    out = []
    while (d := r.next_doc()) is not None:
        out.append(int(d[0]))
    return out


@pytest.mark.parametrize("layout", [[5, 5, 5], [3, 7, 2, 9], [1, 1, 1, 1], [10]])
def test_every_document_in_every_shard_is_read(tmp_path, layout):
    got = _drain(_slice(tmp_path, layout))
    assert got == list(range(1, sum(layout) + 1)), (
        f"layout {layout}: expected {sum(layout)} docs, got {len(got)} — "
        f"missing {sorted(set(range(1, sum(layout)+1)) - set(got))}")


def test_the_specific_regression_shard_one_is_not_skipped(tmp_path):
    """The exact shape of the original failure."""
    got = _drain(_slice(tmp_path, [5, 5, 5]))
    assert len(got) == 15
    assert 6 in got and 10 in got, "documents from shard 1 are missing"


def test_resume_continues_without_gap_or_repeat(tmp_path):
    r = _slice(tmp_path, [4, 4, 4])
    first = [int(r.next_doc()[0]) for _ in range(7)]
    st = r.state()
    r2 = _slice(tmp_path, [4, 4, 4])
    r2.restore(st)
    assert first + _drain(r2) == list(range(1, 13))


@pytest.mark.parametrize("cut", [1, 4, 5, 8, 11])
def test_resume_from_every_boundary(tmp_path, cut):
    r = _slice(tmp_path, [4, 4, 4])
    first = [int(r.next_doc()[0]) for _ in range(cut)]
    r2 = _slice(tmp_path, [4, 4, 4])
    r2.restore(r.state())
    assert first + _drain(r2) == list(range(1, 13)), f"resume at {cut} lost or repeated"


def test_exhaustion_returns_none(tmp_path):
    r = _slice(tmp_path, [2])
    _drain(r)
    assert r.next_doc() is None


def test_state_halves_refer_to_the_same_shard(tmp_path):
    """cur_shard must be the shard doc_i indexes into, not the next one to load."""
    r = _slice(tmp_path, [4, 4])
    for _ in range(5):
        r.next_doc()
    st = r.state()
    assert st["cur_shard"] == 1 and st["doc_i"] == 1


def test_slice_shares_are_a_partition():
    total = sum(share for _d, share, _c in build_mix.SLICES.values())
    assert abs(total - 1.0) < 1e-9, f"slice shares sum to {total}, not 1.0"


def test_only_stack_derived_slices_claim_per_file_attribution():
    """Claiming the-Stack attribution for FineWeb would be a false provenance record."""
    assert set(build_mix.ATTRIBUTION) == {"code", "markdown"}
    assert set(build_mix.DATASET_PROVENANCE) == {"web", "math", "arxiv", "commits"}
    assert set(build_mix.ATTRIBUTION) | set(build_mix.DATASET_PROVENANCE) == \
        set(build_mix.SLICES)
