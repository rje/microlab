"""scripts/build_code_tokenizer_corpora.py cleaning heuristics: minified/generated-blob
detection (the line-length + long-line-fraction rules, with the prose exemption and the
size gate) and non-UTF-8 rejection. These are the pure gates that decide what enters the
corpus, so a regression here silently corrupts the training mix. Loaded via importlib since
scripts/ isn't a package."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "build_code_tokenizer_corpora",
    Path(__file__).resolve().parents[2] / "scripts" / "build_code_tokenizer_corpora.py")
mod = importlib.util.module_from_spec(_SPEC)
# Register before exec so the @dataclass in the module can resolve its own __module__
# (dataclasses look the class's module up in sys.modules).
sys.modules[_SPEC.name] = mod
_SPEC.loader.exec_module(mod)


def test_is_utf8():
    assert mod.is_utf8("normal ascii and é ünïcödé 漢字")
    # lone surrogate -> not clean utf-8
    assert not mod.is_utf8("bad \udc80 surrogate")


def test_minified_single_long_line_dropped():
    # one 1500-char line (no newlines) is the minified-bundle signature
    blob = "a=1;" * 400  # 1600 chars, single line
    assert mod.looks_minified(blob, code=True)


def test_normal_code_kept():
    src = "def add(a, b):\n    # sum two numbers\n    return a + b\n" * 5
    assert not mod.looks_minified(src, code=True)


def test_long_line_fraction_flags_large_dense_file():
    # a large file where most bytes live in >=500-char lines (line cap, no single 1000+ line)
    dense = ("x" * 600 + "\n") * 20  # 20 lines of 600 chars each, total ~12k
    assert mod.looks_minified(dense, code=True)


def test_short_single_long_line_not_flagged_by_fraction():
    # a short single-statement doc (one 600-char SQL query) is below the size gate: kept
    query = "SELECT " + ", ".join(f"col_{i}" for i in range(80)) + " FROM t;\n"
    assert len(query) >= mod.LONG_LINE_CHARS  # it IS a long line
    assert len(query) < mod.LONG_LINE_MIN_TOTAL  # but a small doc
    assert not mod.looks_minified(query, code=True)


def test_prose_long_paragraph_not_minified():
    # unwrapped natural-text paragraph is one long line but must NOT be flagged as prose
    paragraph = ("This is a long unwrapped paragraph of natural English prose that runs on "
                 "for well over five hundred characters without any hard line breaks, which "
                 "is completely normal for crawled web text and must not be mistaken for a "
                 "minified code blob by the cleaner. ") * 4
    assert len(paragraph.split("\n")[0]) >= mod.LONG_LINE_CHARS
    assert not mod.looks_minified(paragraph, code=False)
    # ...but the same rule as code WOULD flag it, proving the exemption matters
    assert mod.looks_minified(paragraph, code=True)


def test_prose_hard_ceiling_still_applies():
    pathological = "x" * (mod.PROSE_MAX_LINE + 1)
    assert mod.looks_minified(pathological, code=False)


def test_empty_is_minified():
    assert mod.looks_minified("", code=True)
    assert mod.looks_minified("", code=False)


def test_clean_documents_reasons_and_normalization():
    docs = [
        "def f():\n    return 1\n" * 4,   # kept
        "a=1;" * 400,                     # minified
        "short",                          # too small (< min_chars)
        "bad \udc80 surrogate " * 10,     # non-utf8
    ]
    results = list(mod.clean_documents(iter(docs), code=True, min_chars=64, max_chars=1_000_000))
    reasons = [r for _, r in results]
    assert reasons[0] is None
    assert reasons[1] == "minified"
    assert reasons[2] == "size"
    assert reasons[3] == "nonutf8"
    # kept doc is newline-terminated
    assert results[0][0].endswith("\n")
