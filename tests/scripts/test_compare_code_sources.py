"""scripts/compare_code_sources.py: the quality-proxy metrics (alphanum fraction,
comment/docstring detection, identifier stats, dup rate), the measurement core over an
injected doc stream (no network), the REUSE of build_code_corpus gates (identity, not a
copy), and the markdown rendering. Loaded via importlib since scripts/ is not a
package."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


build = _load("build_code_corpus")
mod = _load("compare_code_sources")


class CharTok:
    """1 token per character: exact fertility math for tests."""

    def encode_batch(self, docs):
        return [[ord(c) for c in d] for d in docs]


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------


def test_alphanum_fraction():
    assert mod.alphanum_fraction("ab1!?") == pytest.approx(3 / 5)
    with pytest.raises(ValueError):
        mod.alphanum_fraction("")


def test_has_comment_python():
    assert mod.has_comment("x = 1\n# a comment\n", "python")
    assert mod.has_comment('def f():\n    """doc."""\n', "python")
    assert not mod.has_comment("x = 1\ny = 2\n", "python")
    # a '#' mid-line (e.g. in a string) is not a leading comment marker
    assert not mod.has_comment('x = "a#b"\n', "python")


def test_has_comment_js_ts():
    assert mod.has_comment("// hi\nlet x = 1;\n", "javascript")
    assert mod.has_comment("/* block */ const y = 2;\n", "typescript")
    assert not mod.has_comment("const z = 3;\n", "typescript")


def test_has_comment_unknown_language_raises():
    with pytest.raises(ValueError, match="no comment heuristic"):
        mod.has_comment("x", "cobol")


def test_identifier_stats_excludes_keywords():
    n, chars = mod.identifier_stats("def foo_bar(): return baz + qux")
    # def/return are keywords; foo_bar(7) baz(3) qux(3)
    assert n == 3 and chars == 13


def test_dup_rate():
    assert mod.dup_rate([b"a", b"b", b"a"]) == pytest.approx(1 / 3)
    assert mod.dup_rate([b"a", b"b"]) == 0.0
    with pytest.raises(ValueError):
        mod.dup_rate([])


# ---------------------------------------------------------------------------
# Reuse: the comparison must apply the SAME gates as the corpus build
# ---------------------------------------------------------------------------


def test_gates_are_shared_with_corpus_build():
    assert mod.license_ok is build.license_ok
    assert mod.clean_one is build.clean_one
    assert mod._fert is sys.modules.get("tokenizer_fertility")


# ---------------------------------------------------------------------------
# Measurement core over an injected stream
# ---------------------------------------------------------------------------

DOC = "def foo_bar():\n    # add one to the accumulator\n    return baz_qux + 1\n" * 3


def test_measure_pairs():
    minified = "a=1;" * 600  # 2400-char single line: dropped by the reused gate
    pairs = iter([(DOC, ["MIT"]), (DOC, ["GPL-3.0"]), (minified, None)])
    cell = mod.measure_pairs(pairs, lang="python", tok=CharTok(),
                             target_bytes=10**9, quality_bytes=10**6)
    assert cell["streamed_docs"] == 3
    assert cell["kept_docs"] == 2
    assert cell["survival_rate"] == pytest.approx(2 / 3)
    assert cell["drops"] == {"size": 0, "nonutf8": 0, "minified": 1}
    assert cell["dup_rate"] == pytest.approx(1 / 2)  # the two kept docs are identical
    # license metadata existed for 2 docs; only MIT passes the allowlist
    assert cell["license_coverage"] == pytest.approx(1 / 2)
    assert cell["tokens_per_byte"] == pytest.approx(1.0)  # CharTok: 1 token per byte
    assert cell["comment_fraction"] == pytest.approx(1.0)
    # identifiers include comment words: foo_bar(7) add(3) one(3) to(2) the(3)
    # accumulator(11) baz_qux(7) -> 36 chars / 7 idents
    assert cell["mean_ident_len"] == pytest.approx(36 / 7)
    assert 0.0 < cell["alnum_fraction"] < 1.0


def test_measure_pairs_no_license_metadata_reports_none():
    cell = mod.measure_pairs(iter([(DOC, None)]), lang="python", tok=CharTok(),
                             target_bytes=10**9)
    assert cell["license_coverage"] is None


def test_measure_pairs_stops_at_target_bytes():
    pairs = ((DOC, None) for _ in range(10_000))
    cell = mod.measure_pairs(pairs, lang="python", tok=CharTok(),
                             target_bytes=len(DOC.encode()) * 5)
    assert cell["streamed_docs"] == 5


def test_measure_pairs_nothing_survives_raises():
    minified = "a=1;" * 600
    with pytest.raises(ValueError, match="survived"):
        mod.measure_pairs(iter([(minified, None)]), lang="python", tok=CharTok(),
                          target_bytes=10**9)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _cell():
    return {"streamed_docs": 10, "streamed_mb": 1.0, "kept_docs": 9,
            "survival_rate": 0.9, "drops": {"size": 1, "nonutf8": 0, "minified": 0},
            "dup_rate": 0.01, "license_coverage": 0.95, "mean_bytes": 1000.0,
            "p50_bytes": 800.0, "p90_bytes": 2000.0, "mean_lines": 30.0,
            "alnum_fraction": 0.6, "tokens_per_byte": 0.28, "comment_fraction": 0.8,
            "mean_ident_len": 6.5}


def test_render_markdown_with_inaccessible_source():
    results = {
        "generated": "2026-07-29", "mb_per_lang": 200.0, "tokenizer": "code-49k",
        "languages": ["python"],
        "starcoderdata_probe": {"accessible": False, "error": "403", "note": "gated"},
        "sources": {"the-stack-dedup": {"python": _cell()},
                    "substitutes-fresh": {"python": _cell() | {"license_coverage": 0.7}}},
        "notes": ["a note"],
    }
    md = mod.render_markdown(results)
    assert "| the-stack-dedup | 90.0% |" in md
    assert "*not accessible*" in md  # starcoderdata row renders explicitly
    assert "95.0%" in md and "0.280" in md
    assert "## Recommendation" in md
    assert "a note" in md


def test_render_cell_handles_missing_license_coverage():
    assert mod._cell({"license_coverage": None}, "license_coverage", "{:.1%}") == "n/a"
