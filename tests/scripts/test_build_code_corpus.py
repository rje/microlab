"""scripts/build_code_corpus.py: the license allowlist, attribution completeness, the
reuse of the tokenizer-corpus cleaning gates, the ShardDataset-format round trip, and the
checkpoint/resume (cursor) logic. Everything runs against an injected synthetic rows
factory -- no network, no real tokenizer corpus. Loaded via importlib since scripts/ is
not a package."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from microlab.data.shard_dataset import ShardDataset
from microlab.tokenizer.fast import FastTokenizer


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


mod = _load("build_code_corpus")
corpora = _load("build_code_tokenizer_corpora")


# ---------------------------------------------------------------------------
# License allowlist
# ---------------------------------------------------------------------------


def test_license_allowlist_accepts_permissive():
    assert mod.license_ok(["MIT"])
    assert mod.license_ok(["Apache-2.0"])
    assert mod.license_ok(["BSD-3-Clause"])
    assert mod.license_ok(["BSD-2-Clause", "MIT"])  # multi, all permissive
    assert mod.license_ok(["mit"])  # case-insensitive
    assert mod.license_ok([" ISC "])  # whitespace tolerated


def test_license_allowlist_rejects_copyleft_and_unknown():
    assert not mod.license_ok(["GPL-3.0"])
    assert not mod.license_ok(["LGPL-2.1"])
    assert not mod.license_ok(["AGPL-3.0"])
    assert not mod.license_ok(["MPL-2.0"])  # weak copyleft: excluded
    assert not mod.license_ok(["SomeUnknownLicense"])
    assert not mod.license_ok(["MIT", "GPL-2.0"])  # dual-licensed w/ copyleft: rejected
    assert not mod.license_ok([])  # no metadata -> unattributable -> rejected
    assert not mod.license_ok(None)


# ---------------------------------------------------------------------------
# Cleaning reuse (the gates must BE the tokenizer-corpus gates, not a copy)
# ---------------------------------------------------------------------------


def test_cleaning_module_is_shared_not_copied():
    assert mod._corpora is corpora  # same module object -> single source of truth
    assert not hasattr(mod, "looks_minified")  # no duplicated heuristics in the pipeline


def test_clean_one_applies_reused_gates():
    minified = "a=1;" * 600  # single 2400-char line: the minified signature
    assert corpora.looks_minified(minified, code=True)
    _, reason = mod.clean_one(minified)
    assert reason == "minified"
    _, reason = mod.clean_one("x = 1\n")
    assert reason == "size"  # below min_chars
    code = "def add(a, b):\n    # sum\n    return a + b\n" * 5
    text, reason = mod.clean_one(code)
    assert reason is None and text.startswith("def add")


def test_clean_one_null_content_raises():
    with pytest.raises(ValueError, match="null document"):
        mod.clean_one(None)


# ---------------------------------------------------------------------------
# groups_after_skip (parquet row-group cursor math)
# ---------------------------------------------------------------------------


def test_groups_after_skip():
    assert mod.groups_after_skip([10, 10, 10], 0) == ([0, 1, 2], 0)
    assert mod.groups_after_skip([10, 10, 10], 5) == ([0, 1, 2], 5)
    assert mod.groups_after_skip([10, 10, 10], 10) == ([1, 2], 0)
    assert mod.groups_after_skip([10, 10, 10], 25) == ([2], 5)
    assert mod.groups_after_skip([10, 10, 10], 30) == ([], 0)  # whole file consumed
    with pytest.raises(ValueError):
        mod.groups_after_skip([10], -1)


# ---------------------------------------------------------------------------
# End-to-end against a synthetic source
# ---------------------------------------------------------------------------

DOC_TMPL = """def function_{i}(value):
    # doc {i}: compute something mildly interesting
    result = value * {i} + len("padding padding padding")
    if result > 10:
        return result
    return -result
"""


def _make_docs(n: int) -> list[str]:
    return [DOC_TMPL.format(i=i) for i in range(n)]


def _rows_factory_for(docs: list[str], licenses=None, files_of: int = 10**9,
                      row_lang: str = "Python"):
    """Rows factory over synthetic docs; slices files every `files_of` rows so multi-file
    cursors get exercised. Honors (start_file, start_row) like iter_stack_rows."""
    licenses = licenses if licenses is not None else [["MIT"]] * len(docs)

    def factory(lang, *, download_dir, start_file=0, start_row=0):
        for gi, (doc, lic) in enumerate(zip(docs, licenses, strict=True)):
            fi, ri = divmod(gi, files_of)
            if fi < start_file or (fi == start_file and ri < start_row):
                continue
            yield mod.SourceRow(
                content=doc, hexsha=f"hexsha{gi:08d}", repo=f"org/repo{gi % 7}",
                path=f"src/mod_{gi}.py", licenses=lic, lang=row_lang,
                file_idx=fi, row_idx=ri)

    return factory


@pytest.fixture(scope="module")
def tok_path(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("tok") / "tiny.json"
    FastTokenizer.train(_make_docs(50), vocab_size=300, save_path=str(p))
    return p


def _cfg(out: Path, tok_path: Path, **kw) -> object:
    defaults = dict(out=out, tokenizer=str(tok_path), languages=["python"],
                    target_tokens=10**9, val_fraction=0.2, weights=[1.0],
                    shard_size=500, batch_docs=8, checkpoint_rows=25)
    defaults.update(kw)
    return mod.BuildConfig(**defaults)


def test_build_roundtrips_through_sharddataset(tmp_path, tok_path):
    docs = _make_docs(120)
    out = tmp_path / "corpus"
    builder = mod.CorpusBuilder(_cfg(out, tok_path), rows_factory=_rows_factory_for(docs))
    state = builder.run()
    assert state["completed"]

    tok = FastTokenizer.load(str(tok_path))
    total = 0
    for split in ("train", "val"):
        ds = ShardDataset(str(out), split)
        manifest = json.loads((out / f"{split}-manifest.json").read_text())
        assert manifest["dtype"] == "uint16"
        assert ds.total_tokens == manifest["total_tokens"] > 0
        assert [s["file"] for s in manifest["shards"]] == \
            [f"{split}-{i:05d}.bin" for i in range(len(manifest["shards"]))]
        total += ds.total_tokens
    # decoded shard content is the cleaned docs (EOT-separated)
    arr = np.memmap(out / "train-00000.bin", dtype=np.uint16, mode="r")
    decoded = tok.decode([int(t) for t in arr[:200] if int(t) != tok.eot_token])
    assert "def function_" in decoded
    # every token is accounted for in the attribution manifest
    recs = [json.loads(line) for line in (out / "attribution.jsonl").read_text().splitlines()]
    assert sum(r["tokens"] for r in recs) == total
    # copied tokenizer sits alongside the shards (fineweb-100bt layout)
    assert (out / "tokenizer.json").exists()


def test_attribution_records_complete(tmp_path, tok_path):
    docs = _make_docs(40)
    out = tmp_path / "corpus"
    mod.CorpusBuilder(_cfg(out, tok_path), rows_factory=_rows_factory_for(docs)).run()
    recs = [json.loads(line) for line in (out / "attribution.jsonl").read_text().splitlines()]
    assert len(recs) == 40
    for r in recs:
        for key in ("lang", "repo", "hexsha", "path", "licenses", "split", "tokens"):
            assert r[key], (key, r)
        assert r["split"] in ("train", "val")
        assert r["licenses"] == ["MIT"]
    assert len({r["hexsha"] for r in recs}) == 40


def test_license_and_attribution_gates_drop(tmp_path, tok_path):
    docs = _make_docs(30)
    licenses = [["MIT"]] * 10 + [["GPL-3.0"]] * 10 + [[]] * 10
    out = tmp_path / "corpus"
    builder = mod.CorpusBuilder(
        _cfg(out, tok_path), rows_factory=_rows_factory_for(docs, licenses=licenses))
    state = builder.run()
    ls = state["languages"]["python"]
    assert ls["kept_docs"] == 10
    assert ls["dropped_license"] == 20
    recs = [json.loads(line) for line in (out / "attribution.jsonl").read_text().splitlines()]
    assert len(recs) == 10


def test_missing_attribution_metadata_drops_doc(tmp_path, tok_path):
    docs = _make_docs(5)

    def factory(lang, *, download_dir, start_file=0, start_row=0):
        for i, doc in enumerate(docs):
            if i < start_row:
                continue
            yield mod.SourceRow(content=doc, hexsha=f"h{i}" if i != 2 else None,
                                repo="org/r", path=f"p{i}.py", licenses=["MIT"],
                                lang="Python", file_idx=0, row_idx=i)

    state = mod.CorpusBuilder(_cfg(tmp_path / "c", tok_path), rows_factory=factory).run()
    ls = state["languages"]["python"]
    assert ls["dropped_attribution"] == 1 and ls["kept_docs"] == 4


def test_exact_dedup_drops_duplicates(tmp_path, tok_path):
    docs = _make_docs(20) + _make_docs(20)  # every doc twice
    state = mod.CorpusBuilder(
        _cfg(tmp_path / "c", tok_path), rows_factory=_rows_factory_for(docs)).run()
    ls = state["languages"]["python"]
    assert ls["kept_docs"] == 20 and ls["dropped_dup"] == 20


def test_language_partition_mismatch_raises(tmp_path, tok_path):
    factory = _rows_factory_for(_make_docs(3), row_lang="JavaScript")
    with pytest.raises(RuntimeError, match="language partition mismatch"):
        mod.CorpusBuilder(_cfg(tmp_path / "c", tok_path), rows_factory=factory).run()


def test_val_split_deterministic_and_disjoint(tmp_path, tok_path):
    docs = _make_docs(150)
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    for out in (out_a, out_b):
        mod.CorpusBuilder(_cfg(out, tok_path), rows_factory=_rows_factory_for(docs)).run()
    recs_a = [json.loads(x) for x in (out_a / "attribution.jsonl").read_text().splitlines()]
    recs_b = [json.loads(x) for x in (out_b / "attribution.jsonl").read_text().splitlines()]
    assert [r["split"] for r in recs_a] == [r["split"] for r in recs_b]
    assert {r["split"] for r in recs_a} == {"train", "val"}  # 0.2 over 150 docs hits both


# ---------------------------------------------------------------------------
# Resume / cursor logic
# ---------------------------------------------------------------------------


class _Interrupt(Exception):
    pass


def _interrupting(factory, after: int):
    """Wrap a rows factory to raise after yielding `after` rows (simulates a crash)."""

    def wrapped(lang, **kw):
        def gen():
            for i, row in enumerate(factory(lang, **kw)):
                if i >= after:
                    raise _Interrupt()
                yield row
        return gen()

    return wrapped


def test_resume_after_interrupt_matches_uninterrupted(tmp_path, tok_path):
    docs = _make_docs(200)
    factory = _rows_factory_for(docs, files_of=60)  # multi-file cursors

    ref = tmp_path / "ref"
    mod.CorpusBuilder(_cfg(ref, tok_path), rows_factory=factory).run()

    out = tmp_path / "resumed"
    with pytest.raises(_Interrupt):
        mod.CorpusBuilder(_cfg(out, tok_path),
                          rows_factory=_interrupting(factory, 130)).run()
    state = json.loads((out / "build-state.json").read_text())
    assert not state["completed"]
    assert state["languages"]["python"]["rows_done"] > 0

    mod.CorpusBuilder(_cfg(out, tok_path), rows_factory=factory).run()

    for split in ("train", "val"):
        m_ref = json.loads((ref / f"{split}-manifest.json").read_text())
        m_out = json.loads((out / f"{split}-manifest.json").read_text())
        assert m_ref == m_out
        for s in m_ref["shards"]:
            assert (ref / s["file"]).read_bytes() == (out / s["file"]).read_bytes()
    assert (ref / "attribution.jsonl").read_bytes() == (out / "attribution.jsonl").read_bytes()
    assert not list(out.glob("*pending*"))  # checkpoint scratch cleaned up


def test_resume_truncates_post_checkpoint_writes(tmp_path, tok_path):
    """A crash AFTER data writes but BEFORE the state write must roll back cleanly:
    orphan shards, attribution bytes and hashes past the checkpoint are discarded."""
    out = tmp_path / "c"
    docs = _make_docs(80)
    factory = _rows_factory_for(docs)
    with pytest.raises(_Interrupt):
        mod.CorpusBuilder(_cfg(out, tok_path),
                          rows_factory=_interrupting(factory, 60)).run()

    # simulate post-checkpoint writes that the state file never recorded
    with (out / "attribution.jsonl").open("ab") as f:
        f.write(b'{"junk": true}\n')
    manifest = json.loads((out / "train-manifest.json").read_text())
    orphan = f"train-{len(manifest['shards']):05d}.bin"
    np.zeros(500, dtype=np.uint16).tofile(out / orphan)
    manifest["shards"].append({"file": orphan, "tokens": 500})
    manifest["total_tokens"] += 500
    (out / "train-manifest.json").write_text(json.dumps(manifest))

    mod.CorpusBuilder(_cfg(out, tok_path), rows_factory=factory).run()
    if (out / orphan).exists():  # name may be legitimately reused by the resumed build
        arr = np.fromfile(out / orphan, dtype=np.uint16)
        assert arr.any()  # ... but the all-zeros orphan content must be gone
    text = (out / "attribution.jsonl").read_text()
    assert "junk" not in text
    recs = [json.loads(x) for x in text.splitlines()]
    assert len(recs) == 80  # exactly one record per kept doc, no dup, no loss
    # state offsets match reality
    final = json.loads((out / "build-state.json").read_text())
    assert final["completed"]
    assert (out / "attribution.jsonl").stat().st_size == final["attribution_bytes"]


def test_resume_with_changed_config_refuses(tmp_path, tok_path):
    out = tmp_path / "c"
    factory = _rows_factory_for(_make_docs(60))
    with pytest.raises(_Interrupt):
        mod.CorpusBuilder(_cfg(out, tok_path),
                          rows_factory=_interrupting(factory, 40)).run()
    with pytest.raises(RuntimeError, match="config mismatch"):
        mod.CorpusBuilder(_cfg(out, tok_path, target_tokens=123), rows_factory=factory)


def test_fresh_build_refuses_foreign_dir(tmp_path, tok_path):
    out = tmp_path / "c"
    out.mkdir()
    (out / "train-manifest.json").write_text("{}")
    with pytest.raises(FileExistsError, match="foreign corpus dir"):
        mod.CorpusBuilder(_cfg(out, tok_path), rows_factory=_rows_factory_for([]))


def test_completed_build_is_a_noop_on_rerun(tmp_path, tok_path):
    out = tmp_path / "c"
    factory = _rows_factory_for(_make_docs(30))
    mod.CorpusBuilder(_cfg(out, tok_path), rows_factory=factory).run()
    before = (out / "train-manifest.json").read_bytes()
    mod.CorpusBuilder(_cfg(out, tok_path), rows_factory=factory).run()
    assert (out / "train-manifest.json").read_bytes() == before


def test_near_dedup_hook_is_explicitly_unimplemented():
    assert mod.near_dup_reason("any text") is None  # keeps everything today
    assert "MinHash" in mod.near_dup_reason.__doc__


def test_markdown_is_gated_as_prose_not_code():
    """A README paragraph is one long unwrapped line; the CODE rules call that minified.

    Markdown entered the mix for the docs slice, and routing it through the code path
    would silently drop ordinary prose files.
    """
    b = _load("build_code_corpus")
    c = _load("build_code_tokenizer_corpora")

    para = "This project provides a fast, dependency-free parser. " * 40  # ~2.1k chars
    doc = f"# Title\n\n{para}\n"
    assert "markdown" in b.PROSE_LANGS
    assert c.looks_minified(doc, code=True) is True     # would be dropped as code
    assert c.looks_minified(doc, code=False) is False   # kept as prose


def test_markdown_partition_is_registered():
    b = _load("build_code_corpus")
    assert b.STACK_DIRS["markdown"] == "data/markdown"
    assert b.STACK_LANG_NAMES["markdown"] == "Markdown"
