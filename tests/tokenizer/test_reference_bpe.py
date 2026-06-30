
from microlab.tokenizer.reference.bpe import BPETokenizer


def test_roundtrip_arbitrary_unicode():
    tok = BPETokenizer()
    tok.train("the quick brown fox jumps over the lazy dog. " * 20, vocab_size=300)
    for s in ["hello world", "café déjà vu — naïve", "新しい", "a\nb\tc", ""]:
        assert tok.decode(tok.encode(s)) == s


def test_roundtrip_without_training_is_identity_bytes():
    tok = BPETokenizer()  # no merges -> ids are raw bytes
    s = "round trip!"
    assert tok.encode(s) == list(s.encode("utf-8"))
    assert tok.decode(tok.encode(s)) == s


def test_vocab_grows_to_target():
    # A rich, varied corpus sustains enough distinct pairs to reach the target vocab.
    # (A tiny periodic corpus would correctly exhaust early — nothing left to merge.)
    corpus = (
        "the quick brown fox jumps over the lazy dog. "
        "pack my box with five dozen liquor jugs. "
    ) * 40
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=320)
    assert len(tok.vocab) == 320
    assert len(tok.merges) == 320 - 256


def test_known_first_merge():
    # In "aaabdaaabac"*N the most common adjacent pair is (97,97) = "aa".
    tok = BPETokenizer()
    tok.train("aaabdaaabac" * 50, vocab_size=257)
    assert (97, 97) in tok.merges
    assert tok.merges[(97, 97)] == 256


def test_training_is_deterministic():
    a, b = BPETokenizer(), BPETokenizer()
    corpus = "the cat sat on the mat. the cat ran. " * 30
    a.train(corpus, vocab_size=320)
    b.train(corpus, vocab_size=320)
    assert a.merges == b.merges


def test_encode_uses_learned_merges():
    tok = BPETokenizer()
    tok.train("aaab" * 100, vocab_size=258)
    # "aa" merged -> encoding "aaaa" should be shorter than 4 tokens
    assert len(tok.encode("aaaa")) < 4
