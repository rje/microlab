"""Assertions about the BUILT CORPUS, not about the builder.

Every check the repo had ran against the builder's algorithm or against token counts, and
all of them passed while the held-out set was a single repository of geological fault-mesh
data whose remaining files were in train. Counts, manifests and mix proportions were all
exactly right; the val set was still useless, and it was the only number the run reported
between milestones.

The lesson is about WHERE the assertion lives. A builder test says "the deficit algorithm
converges". An artifact test says "the thing on disk has the properties the experiment
needs". Only the second kind could have caught this, and the same class of bug — a data
job emitting structurally valid, semantically wrong output — has now happened three times
in this repo.

These run against a real mix directory when one is present and skip otherwise, so they are
cheap in CI and decisive before a paid run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
# Overridable so the same assertions can be pointed at a freshly built corpus before
# it replaces the live one.
MIX = Path(os.environ.get("MICROLAB_MIX_DIR", ROOT / "data" / "shards" / "mix-v1"))
pytestmark = pytest.mark.corpus

SLICES = {"code", "web", "math", "markdown", "arxiv", "commits"}
BLOCK = 32_768                      # configs/coder-1b.py block_size


def _state():
    p = MIX / "mix-state-train.json"
    if not p.exists():
        pytest.skip(f"no built mix at {MIX}")
    return json.loads(p.read_text())


def _manifest(split: str):
    p = MIX / f"{split}-manifest.json"
    if not p.exists():
        pytest.skip(f"no {split} manifest at {MIX}")
    return json.loads(p.read_text())


def test_val_contains_every_slice():
    """The bug, stated as a property.

    val was 100% code because the deficit was computed from the TRAIN counters while the
    val half was being filled: `written` is 0 throughout that phase, so every slice's
    deficit collapsed to its raw weight and code — permanently the largest — won every
    draw. Nothing downstream could tell, because the token counts were all correct.
    """
    v = _state().get("per_slice_val", {})
    present = {k for k, n in v.items() if n > 0}
    assert present == SLICES, (
        f"val is missing {sorted(SLICES - present)}; a val set that is one slice cannot "
        f"detect that slice diverging, which is the failure it exists to catch. Got {v}")


def test_val_composition_tracks_the_target_mix():
    """A val set that is not distributed like train measures the wrong thing."""
    v = _state().get("per_slice_val", {})
    total = sum(v.values())
    if total == 0:
        pytest.skip("no val tokens recorded")
    targets = {"code": 0.663, "web": 0.150, "math": 0.100,
               "markdown": 0.050, "arxiv": 0.025, "commits": 0.012}
    for name, want in targets.items():
        got = v.get(name, 0) / total
        assert abs(got - want) < 0.05, \
            f"val {name} share {got:.3f} vs target {want:.3f}"


def test_val_is_not_one_giant_document():
    """The broken val held ~1,400 documents from ONE repository. Real held-out data drawn
    across six slices has many more, and far shorter, documents."""
    man = _manifest("val")
    first = MIX / man["shards"][0]["file"]
    if not first.exists():
        pytest.skip("val shard not present locally")
    a = np.fromfile(first, dtype=np.uint16, count=20_000_000)
    # Density, not a raw count, so the threshold holds for a val set of any size. The
    # broken val ran ~35,000 tokens per document (whole packed repositories); a genuine
    # six-slice mix is dominated by web and markdown documents of ~1k.
    per_doc = len(a) / max(int((a == 0).sum()), 1)
    assert per_doc < 15_000, (
        f"{per_doc:,.0f} tokens per document in val — that is the signature of a val set "
        f"drawn from whole packed repositories rather than across the six slices")


def test_manifest_token_counts_are_self_consistent():
    for split in ("train", "val"):
        man = _manifest(split)
        assert sum(s["tokens"] for s in man["shards"]) == man["total_tokens"], \
            f"{split} manifest total disagrees with its shard sum"


def test_no_shard_is_shorter_than_a_training_window():
    """sequence_at needs block_size+1 tokens; a short shard raises mid-run."""
    for split in ("train", "val"):
        for s in _manifest(split)["shards"]:
            assert s["tokens"] > BLOCK + 1, f"{s['file']} holds only {s['tokens']:,} tokens"


def test_fim_examples_fit_inside_a_training_window():
    """73.8% of FIM tokens were once in documents longer than the window, so the triple
    could never be seen whole — roughly a quarter of the corpus teaching an unlearnable
    format. FIM is now applied per chunk; spans must stay inside one window."""
    man = _manifest("val")
    first = MIX / man["shards"][0]["file"]
    if not first.exists():
        pytest.skip("val shard not present locally")
    tokp = MIX / "tokenizer.json"
    if not tokp.exists():
        pytest.skip("no tokenizer beside the mix")
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(str(tokp))
    pre = t.token_to_id("<|fim_prefix|>")
    mid = t.token_to_id("<|fim_middle|>")
    a = np.fromfile(first, dtype=np.uint16)
    starts = np.flatnonzero(a == pre)
    mids = np.flatnonzero(a == mid)
    if len(starts) < 2 or len(mids) < 2:
        pytest.skip("too few FIM examples in this shard to measure")
    # Distance from each prefix sentinel to ITS middle sentinel — that is the span the
    # model must hold in one window to see prefix and suffix before predicting the middle.
    # NOT the gap between consecutive prefixes: with a 0.5 rate and five non-code slices
    # interleaved, long gaps between examples are expected and harmless.
    j = np.searchsorted(mids, starts, side="left")
    ok = j < len(mids)
    reach = mids[j[ok]] - starts[ok]
    assert reach.max() <= BLOCK, (
        f"longest FIM prefix->middle reach is {int(reach.max()):,} tokens against a "
        f"{BLOCK:,} window — no training window can contain the triple together")
