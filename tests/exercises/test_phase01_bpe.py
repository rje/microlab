"""Spec + validation for the hand-written byte-level BPE.

Implement `microlab.exercises.phase01_bpe.BPETokenizer` until these pass. The property tests
(round-trip, determinism, vocab size) are tie-break-agnostic; the final differential
test pins you to the reference oracle's exact merges, so match its tie-break
(`(count, -a, -b)` argmax — larger byte values win ties).
"""

import pytest

from microlab.data.reference.loaders import load_sample
from microlab.exercises.phase01_bpe import BPETokenizer

# Varied public-domain prose — rich enough to sustain the merges these tests need.
_CORPUS = load_sample()


def test_roundtrip_arbitrary_unicode():
    tok = BPETokenizer()
    tok.train(_CORPUS, vocab_size=320)
    for s in ["hello world", "café déjà vu — naïve", "新しい", "a\nb\tc", ""]:
        assert tok.decode(tok.encode(s)) == s


def test_untrained_is_identity_bytes():
    tok = BPETokenizer()
    s = "round trip!"
    assert tok.encode(s) == list(s.encode("utf-8"))
    assert tok.decode(tok.encode(s)) == s


def test_vocab_grows_to_target():
    tok = BPETokenizer()
    tok.train(_CORPUS, vocab_size=320)
    assert len(tok.vocab) == 320
    assert len(tok.merges) == 320 - 256


def test_training_is_deterministic():
    a, b = BPETokenizer(), BPETokenizer()
    a.train(_CORPUS, vocab_size=320)
    b.train(_CORPUS, vocab_size=320)
    assert a.merges == b.merges


def test_encode_uses_learned_merges():
    tok = BPETokenizer()
    tok.train("aaab" * 100, vocab_size=258)
    assert len(tok.encode("aaaa")) < 4


def test_differential_vs_reference_oracle():
    from microlab.tokenizer.reference.bpe import BPETokenizer as Reference

    mine, ref = BPETokenizer(), Reference()
    mine.train(_CORPUS, vocab_size=400)
    ref.train(_CORPUS, vocab_size=400)
    assert mine.merges == ref.merges
    assert mine.vocab == ref.vocab
    for s in ["the quick fox", "five dozen liquor jugs", "unseen WORDS 123"]:
        assert mine.encode(s) == ref.encode(s)

pytestmark = pytest.mark.exercise
