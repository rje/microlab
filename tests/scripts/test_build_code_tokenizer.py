"""scripts/build_code_tokenizer.py: the code-tokenizer recipe (digit-splitting byte-level
BPE), the 60/30/10 byte-budget allocator, indentation-merge detection, and an end-to-end
train_candidate on a tiny synthetic corpus. Verifies the two documented deficiencies are
addressed: digits split to individual tokens, and multi-space indentation merges are learned.
Loaded via importlib since scripts/ isn't a package."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from tokenizers import pre_tokenizers, trainers

_SPEC = importlib.util.spec_from_file_location(
    "build_code_tokenizer",
    Path(__file__).resolve().parents[2] / "scripts" / "build_code_tokenizer.py")
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)

from microlab.tokenizer.fast import EOT, FastTokenizer  # noqa: E402


def _train_inline(vocab_size=1000):
    corpus = [
        "def f(x):\n    return x + 12345\n" * 30,
        "for i in range(100):\n    total = i * 2 + 3.14159\n" * 30,
        "const hex = 0xFF3A; let n = 1000000;\n" * 30,
        "the value 42 and pi 3.14159 recur in prose 255 times\n" * 30,
    ]
    tok = mod.build_code_tokenizer(vocab_size)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[EOT],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(corpus, trainer=trainer)
    return tok


def test_digits_split_to_individual_tokens():
    tok = _train_inline()
    ft = FastTokenizer(tok)
    # the core requirement: every digit is its own token
    assert len(ft.encode("123")) == 3
    assert len(ft.encode("12345")) == 5
    assert len(ft.encode("1000000")) == 7
    # "3.14159" -> 3 . 1 4 1 5 9 = 7 tokens
    assert len(ft.encode("3.14159")) == 7


def test_recipe_roundtrips():
    ft = FastTokenizer(_train_inline())
    for s in ["def f(x):\n    return x", "0xFF3A", "12345", "\ttabbed\n    spaced"]:
        assert ft.decode(ft.encode(s)) == s


def test_multispace_tokens_learned():
    # heavy 4-space indentation should induce at least one multi-space merge token
    tok = _train_inline()
    assert mod.count_multispace_tokens(tok, min_spaces=2) >= 1


def test_plan_byte_budget_hits_bucket_weights_unclamped():
    available = {lang: 10**9 for lang in
                 ("python", "javascript", "typescript", "shell", "sql", "json",
                  "markdown", "prose")}
    plan = mod.plan_byte_budget(available, 1_000_000)
    # code 60% / 3 langs, glue 10% / 4 langs, prose 30% / 1 lang
    assert plan["python"] == plan["javascript"] == plan["typescript"] == 200_000
    assert plan["shell"] == plan["sql"] == plan["json"] == plan["markdown"] == 25_000
    assert plan["prose"] == 300_000


def test_plan_byte_budget_clamps_to_available():
    available = {lang: 1000 for lang in
                 ("python", "javascript", "typescript", "shell", "sql", "json",
                  "markdown", "prose")}
    plan = mod.plan_byte_budget(available, 10_000)
    # code wants 2000/lang and prose 3000, both clamped to the 1000 available
    assert plan["python"] == 1000
    assert plan["prose"] == 1000
    # glue wants 250/lang (< available), so unclamped
    assert plan["shell"] == 250


def test_bucket_of_rejects_unknown():
    with pytest.raises(ValueError):
        mod._bucket_of("rust")


def _write_corpus(root: Path):
    blocks = {
        "python": "def f(x):\n    if x > 10:\n        return x + 12345\n    return 3.14159\n",
        "javascript": "function g(y) {\n    const z = 0xFF3A;\n    return z + 1000;\n}\n",
        "typescript": "const h = (n: number): number => {\n    return n * 255;\n};\n",
        "shell": "#!/bin/sh\nfor i in 1 2 3; do\n    echo \"$i done\"\ndone\n",
        "sql": "SELECT id, name FROM users WHERE age > 21 ORDER BY id;\n",
        "json": '{\n    "id": 42,\n    "items": [1, 2, 3],\n    "ok": true\n}\n',
        "markdown": "# Title\n\nSome **bold** text and a list:\n\n- one\n- two\n",
        "prose": "The quick brown fox jumps over the lazy dog many times in this text.\n",
    }
    for lang, block in blocks.items():
        d = root / lang
        d.mkdir(parents=True)
        (d / "sample-0000.txt").write_text((block + "\n") * 200, encoding="utf-8")


def test_train_candidate_end_to_end(tmp_path):
    root = tmp_path / "corpora"
    _write_corpus(root)
    out = tmp_path / "tok" / "code-test.json"
    ft = mod.train_candidate(root, out, vocab_size=800, total_budget=2_000_000)
    assert out.exists()
    assert ft.vocab_size <= 800
    # digit split + round-trip enforced inside train_candidate; re-check from disk
    loaded = FastTokenizer.load(str(out))
    assert len(loaded.encode("123")) == 3
    assert loaded.decode(loaded.encode("def f():\n    return 1")) == "def f():\n    return 1"


def test_train_candidate_raises_on_missing_language(tmp_path):
    root = tmp_path / "corpora"
    (root / "python").mkdir(parents=True)
    (root / "python" / "s.txt").write_text("def f():\n    return 1\n" * 50)
    with pytest.raises(FileNotFoundError):
        mod.train_candidate(root, tmp_path / "t.json", vocab_size=500, total_budget=1000)
