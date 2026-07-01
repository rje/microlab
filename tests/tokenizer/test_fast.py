from microlab.tokenizer.fast import FastTokenizer

_CORPUS = [
    "the quick brown fox jumps over the lazy dog. " * 20,
    "pack my box with five dozen liquor jugs. " * 20,
    "machine learning models are trained on large text corpora. " * 20,
]


def test_trains_and_roundtrips():
    tok = FastTokenizer.train(_CORPUS, vocab_size=1000)
    for s in ["the quick brown fox", "machine learning", "hello world 123"]:
        assert tok.decode(tok.encode(s)) == s


def test_vocab_size_and_compression():
    tok = FastTokenizer.train(_CORPUS, vocab_size=1000)
    assert tok.vocab_size <= 1000
    text = "the quick brown fox jumps over the lazy dog"
    ids = tok.encode(text)
    assert len(ids) < len(text)  # compresses vs one-token-per-char
    assert len(text.encode()) / len(ids) > 1.5  # decent bytes/token


def test_save_and_load(tmp_path):
    p = str(tmp_path / "tok.json")
    tok = FastTokenizer.train(_CORPUS, vocab_size=1000, save_path=p)
    loaded = FastTokenizer.load(p)
    assert loaded.encode("the quick brown fox") == tok.encode("the quick brown fox")
    assert loaded.eot_token is not None


def test_roundtrip_via_reference_corpus():
    # trains fine on the bundled sample too (sanity that it handles real prose)
    from microlab.data.reference.loaders import load_sample
    tok = FastTokenizer.train([load_sample()], vocab_size=2000)
    s = "The fox and the grapes"
    assert tok.decode(tok.encode(s)) == s
