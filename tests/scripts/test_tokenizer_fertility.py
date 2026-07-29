"""scripts/tokenizer_fertility.py: fertility math (tokens/bytes/lines aggregation and the
derived ratios), round-trip fraction, digit/indent probes, and corpus reading. The math is
checked against a deterministic char-level tokenizer so exact counts are known; the digit
probe is also checked on a real digit-splitting tokenizer. Loaded via importlib since
scripts/ isn't a package."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from tokenizers import pre_tokenizers, trainers

_SPEC = importlib.util.spec_from_file_location(
    "tokenizer_fertility",
    Path(__file__).resolve().parents[2] / "scripts" / "tokenizer_fertility.py")
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)

from microlab.tokenizer.fast import EOT, FastTokenizer  # noqa: E402


class CharTok:
    """Deterministic 1-token-per-character tokenizer for exact fertility math."""

    def encode(self, s):
        return [ord(c) for c in s]

    def encode_batch(self, docs):
        return [self.encode(d) for d in docs]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def test_fertility_math_exact():
    tok = CharTok()
    docs = ["abc\ndef", "12345\n"]  # 7 chars/1 partial-last-line ; 6 chars/1 line
    f = mod.fertility_of_docs(tok, docs)
    # char tok: tokens == chars == bytes here (all ascii)
    assert f["tokens"] == 7 + 6
    assert f["bytes"] == 7 + 6
    # "abc\ndef" has no trailing newline -> 2 lines; "12345\n" ends in newline -> 1 line
    assert f["lines"] == 2 + 1
    assert f["tokens_per_byte"] == f["tokens"] / f["bytes"]
    assert f["bytes_per_token"] == f["bytes"] / f["tokens"]
    assert f["tokens_per_line"] == f["tokens"] / f["lines"]


def test_roundtrip_fraction():
    tok = CharTok()
    assert mod.roundtrip_fraction(tok, ["hello", "world\n123"]) == 1.0


def test_indent_cost_char_tokenizer():
    tok = CharTok()
    cost = mod.indent_cost(tok, widths=(2, 4, 8))
    # char tok spends one token per leading space
    assert cost["spaces_tokens"] == {2: 2, 4: 4, 8: 8}
    assert cost["nested_snippet_tokens"] == len(mod.NESTED_PY)


def _digit_splitting_tok():
    corpus = ["value 12345 and 3.14159 and 0xFF3A repeated\n" * 40,
              "def f(x):\n    return x + 100\n" * 40]
    spec = importlib.util.spec_from_file_location(
        "build_code_tokenizer",
        Path(__file__).resolve().parents[2] / "scripts" / "build_code_tokenizer.py")
    btk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(btk)
    tok = btk.build_code_tokenizer(1000)
    trainer = trainers.BpeTrainer(vocab_size=1000, special_tokens=[EOT],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(corpus, trainer=trainer)
    return FastTokenizer(tok)


def test_digit_probe_marks_individual_split():
    ft = _digit_splitting_tok()
    probes = mod.digit_probe(ft, probes=["12345", "3.14159", "255"])
    by_text = {p["text"]: p for p in probes}
    assert by_text["12345"]["tokens"] == 5
    assert by_text["12345"]["digits_individually_split"] is True
    assert by_text["255"]["digits_individually_split"] is True
    assert all(p["roundtrip"] for p in probes)


def test_read_docs_splits_and_caps(tmp_path):
    d = tmp_path / "lang"
    d.mkdir()
    (d / "a.txt").write_text("doc-one body\n\ndoc-two body\n\ndoc-three body\n",
                             encoding="utf-8")
    docs = mod.read_docs(d, max_bytes=10**9)
    assert docs == ["doc-one body", "doc-two body", "doc-three body"]
    # byte cap stops early (first doc already exceeds a tiny cap)
    capped = mod.read_docs(d, max_bytes=1)
    assert capped == ["doc-one body"]
