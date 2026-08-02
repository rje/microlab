"""FIM PSM transform: the round trip is the correctness bar.

If `defim(fim_transform(d)) != d` then the transform is losing or reordering code, and we
would be pretraining on documents whose content we cannot account for.
"""

from __future__ import annotations

import numpy as np
import pytest
from tokenizers import Tokenizer

from microlab.data.fim import (
    FIM_TOKENS,
    FIMConfig,
    defim,
    fim_transform,
    split_documents,
)

TOKENIZER = "data/tokenizers/code-49k-fim.json"


@pytest.fixture(scope="module")
def cfg():
    return FIMConfig(Tokenizer.from_file(TOKENIZER))


def test_tokenizer_has_the_sentinels(cfg):
    assert cfg.prefix != cfg.suffix != cfg.middle
    assert len({cfg.prefix, cfg.suffix, cfg.middle}) == 3


def test_missing_sentinels_raise_rather_than_skip():
    """Silently skipping FIM would yield a corpus that looks fine and teaches nothing."""
    plain = Tokenizer.from_file("data/tokenizers/code-49k.json")
    with pytest.raises(ValueError, match="FIM sentinels"):
        FIMConfig(plain)


@pytest.mark.parametrize("seed", range(25))
def test_round_trip_recovers_the_document_exactly(cfg, seed):
    rng = np.random.default_rng(seed)
    doc = rng.integers(1, 40000, size=int(rng.integers(4, 200))).tolist()
    assert defim(fim_transform(doc, cfg, rng), cfg) == doc


def test_transform_actually_reorders(cfg):
    """A no-op that round-trips would pass the test above and teach no infilling."""
    rng = np.random.default_rng(0)
    doc = list(range(1, 60))
    out = fim_transform(doc, cfg, rng)
    assert out[0] == cfg.prefix
    assert cfg.suffix in out and cfg.middle in out
    assert out != doc
    # the middle must be non-empty, or we train "infill -> nothing"
    assert len(out[out.index(cfg.middle) + 1:]) > 0


def test_short_documents_pass_through(cfg):
    rng = np.random.default_rng(0)
    for d in ([], [5], [5, 6], [5, 6, 7]):
        assert fim_transform(d, cfg, rng) == d


def test_defim_passes_through_non_fim_documents(cfg):
    assert defim([1, 2, 3], cfg) == [1, 2, 3]


def test_defim_raises_on_malformed(cfg):
    with pytest.raises(ValueError, match="malformed FIM"):
        defim([cfg.prefix, 1, 2, 3], cfg)          # no suffix/middle markers


def test_split_documents_on_eot():
    toks = np.array([1, 2, 0, 3, 4, 5, 0, 6], dtype=np.uint16)
    docs = split_documents(toks, eot=0)
    assert [d.tolist() for d in docs] == [[1, 2], [3, 4, 5], [6]]


def test_split_documents_skips_empty_runs():
    toks = np.array([0, 0, 1, 0], dtype=np.uint16)
    assert [d.tolist() for d in split_documents(toks, eot=0)] == [[1]]


def test_fim_ids_are_outside_the_base_vocab(cfg):
    """Existing shards hold ids < 49152; the sentinels must not collide with them."""
    assert min(cfg.prefix, cfg.suffix, cfg.middle) >= 49152


def test_sentinel_names_are_the_cohort_standard():
    assert FIM_TOKENS == ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>")
