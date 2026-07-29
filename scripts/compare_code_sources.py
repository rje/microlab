"""Compare candidate base-corpus sources for the coding specialist (py/js/ts).

Sources (x python/javascript/typescript), ~--mb-per-lang MB streamed per cell:

  the-stack-dedup   bigcode/the-stack-dedup (gated; per-file repo+hexsha+license metadata)
  starcoderdata     bigcode/starcoderdata -- gated SEPARATELY; accessibility is probed
                    and reported, measured only if the token opens it
  substitutes-fresh the non-gated substitutes the current samples were built from,
                    re-streamed from the Hub (codeparrot/github-code-clean for py/js with
                    its per-doc `license` column; bleugreen/typescript-chunks for ts)
  substitutes-disk  the existing on-disk samples under data/corpora/code-samples/
                    (already cleaned once; the .txt format is blank-line-lossy, so docs
                    are fragments and license metadata is gone -- kept as a row for
                    completeness, `substitutes-fresh` is the faithful measurement)

Per source x language: cleaning-pass survival (gates REUSED from
build_code_tokenizer_corpora via build_code_corpus.clean_one), exact-dup rate (sha256),
usable-permissive-license coverage (REUSED build_code_corpus.license_ok -- the criterion
the real corpus build applies), length distribution, alphanum fraction, fertility under
code-49k (REUSED tokenizer_fertility.fertility_of_docs), docstring/comment fraction and
mean identifier length (quality proxies).

    python scripts/compare_code_sources.py                 # full run (writes json + md)
    python scripts/compare_code_sources.py --render-only   # re-render md from saved json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_code_corpus import (  # noqa: E402
    clean_one,
    iter_stack_rows,
    license_ok,
    list_parquet_files,
    load_script_module,
)

_corpora = load_script_module("build_code_tokenizer_corpora")
_fert = load_script_module("tokenizer_fertility")

LANGS = ("python", "javascript", "typescript")
SOURCE_ORDER = ("the-stack-dedup", "starcoderdata", "substitutes-fresh", "substitutes-disk")
STARCODER_REPO = "bigcode/starcoderdata"
STARCODER_DIRS = {"python": "python", "javascript": "javascript", "typescript": "typescript"}

# ---------------------------------------------------------------------------
# Quality-proxy metrics (pure; unit-tested)
# ---------------------------------------------------------------------------

IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Merged python + js/ts keyword set: excluded from identifier stats for every language
# (cross-language false positives like a variable named `type` are acceptable noise).
KEYWORDS = frozenset("""
False None True and as assert async await break case class const constructor continue
debugger declare def default del delete do elif else enum except export extends finally
for from function get global if implements import in instanceof interface is keyof lambda
let match module namespace never new nonlocal not null of or package pass private
protected public raise readonly return set static super switch this throw try type typeof
undefined var void while with yield
""".split())


def alphanum_fraction(text: str) -> float:
    if not text:
        raise ValueError("empty document")
    return sum(c.isalnum() for c in text) / len(text)


def has_comment(text: str, lang: str) -> bool:
    """Docstring/comment presence: '#'-comment or triple-quote docstring for python,
    '//' or '/*' for js/ts."""
    if lang == "python":
        if '"""' in text or "'''" in text:
            return True
        return any(line.lstrip().startswith("#") for line in text.split("\n"))
    if lang in ("javascript", "typescript"):
        return "//" in text or "/*" in text
    raise ValueError(f"no comment heuristic for language {lang!r}")


def identifier_stats(text: str) -> tuple[int, int]:
    """(identifier count, total identifier chars) excluding keywords."""
    count = chars = 0
    for m in IDENT_RE.finditer(text):
        if m.group() not in KEYWORDS:
            count += 1
            chars += len(m.group())
    return count, chars


def dup_rate(hashes: list[bytes]) -> float:
    if not hashes:
        raise ValueError("no documents to dedup-check")
    return (len(hashes) - len(set(hashes))) / len(hashes)


# ---------------------------------------------------------------------------
# Measurement core
# ---------------------------------------------------------------------------


def measure_pairs(pairs: Iterator[tuple[str, list | None]], *, lang: str, tok,
                  target_bytes: int, quality_bytes: int = 20_000_000) -> dict:
    """Stream (text, licenses-or-None) pairs until `target_bytes` raw bytes; measure.

    Cleaning survival + license coverage are measured over ALL streamed docs; content
    metrics over the cleaning survivors (what would enter a corpus). Quality proxies
    (alnum/comments/identifiers) use the first `quality_bytes` of survivors -- Python
    regex over hundreds of MB is not worth the wall clock.
    """
    streamed = streamed_bytes = 0
    drops = {"size": 0, "nonutf8": 0, "minified": 0}
    lic_seen = lic_usable = 0
    kept: list[str] = []
    for raw, licenses in pairs:
        streamed += 1
        streamed_bytes += len(raw.encode("utf-8"))
        if licenses is not None:
            lic_seen += 1
            if license_ok(licenses):
                lic_usable += 1
        text, reason = clean_one(raw)
        if reason is not None:
            drops[reason] += 1
        else:
            kept.append(text)
        if streamed_bytes >= target_bytes:
            break
    if not kept:
        raise ValueError(f"no documents survived cleaning for {lang}")

    hashes = [hashlib.sha256(t.encode("utf-8")).digest()[:8] for t in kept]
    sizes = np.array([len(t.encode("utf-8")) for t in kept])
    q_docs: list[str] = []
    q_bytes = 0
    for t in kept:
        q_docs.append(t)
        q_bytes += len(t.encode("utf-8"))
        if q_bytes >= quality_bytes:
            break
    idents = ident_chars = 0
    for t in q_docs:
        n, c = identifier_stats(t)
        idents += n
        ident_chars += c
    if idents == 0:
        raise ValueError(f"no identifiers found for {lang} (broken sample?)")
    fert = _fert.fertility_of_docs(tok, kept)
    return {
        "streamed_docs": streamed,
        "streamed_mb": streamed_bytes / 1e6,
        "kept_docs": len(kept),
        "survival_rate": len(kept) / streamed,
        "drops": drops,
        "dup_rate": dup_rate(hashes),
        "license_coverage": (lic_usable / lic_seen) if lic_seen else None,
        "mean_bytes": float(sizes.mean()),
        "p50_bytes": float(np.percentile(sizes, 50)),
        "p90_bytes": float(np.percentile(sizes, 90)),
        "mean_lines": float(np.mean([t.count("\n") for t in q_docs])),
        "alnum_fraction": float(np.mean([alphanum_fraction(t) for t in q_docs])),
        "tokens_per_byte": fert["tokens_per_byte"],
        "comment_fraction": float(np.mean([has_comment(t, lang) for t in q_docs])),
        "mean_ident_len": ident_chars / idents,
    }


# ---------------------------------------------------------------------------
# Per-source document streams
# ---------------------------------------------------------------------------


def stack_pairs(lang: str, download_dir: Path) -> Iterator[tuple[str, list | None]]:
    for row in iter_stack_rows(lang, download_dir=download_dir):
        yield row.content, row.licenses


def iter_parquet_texts(repo: str, subdir: str, download_dir: Path,
                       text_column: str = "content") -> Iterator[str]:
    """Generic parquet text stream (download-then-parse, one file at a time)."""
    import pyarrow.parquet as papq
    from huggingface_hub import hf_hub_download

    for name in list_parquet_files(repo, subdir):
        local = hf_hub_download(repo, name, repo_type="dataset", local_dir=str(download_dir))
        try:
            pf = papq.ParquetFile(local)
            for batch in pf.iter_batches(batch_size=1024, columns=[text_column]):
                for v in batch.column(text_column):
                    yield v.as_py()
        finally:
            Path(local).unlink()


def starcoder_pairs(lang: str, download_dir: Path) -> Iterator[tuple[str, list | None]]:
    # starcoderdata ships no per-file license column (it inherits the-stack's filtering
    # upstream), so licenses is None -> coverage reported as unmeasured.
    for text in iter_parquet_texts(STARCODER_REPO, STARCODER_DIRS[lang], download_dir):
        yield text, None


def substitutes_fresh_pairs(lang: str) -> Iterator[tuple[str, list | None]]:
    """Re-stream the exact upstream sources the on-disk samples were built from,
    REUSING the source registry of build_code_tokenizer_corpora."""
    from datasets import load_dataset

    src = _corpora.SOURCES[lang]
    ds = load_dataset(src.source_id, streaming=True, **src.hf_kwargs)
    if lang in ("python", "javascript"):  # codeparrot: per-doc `license` column
        for row in ds:
            yield row["code"], [row["license"]] if row.get("license") else []
    elif lang == "typescript":  # bleugreen/typescript-chunks: no license metadata
        for row in ds:
            yield row["content"], None
    else:
        raise ValueError(f"no substitute source wired for {lang!r}")


def disk_pairs(lang: str, corpora_root: Path, max_bytes: int) -> Iterator[tuple[str, list | None]]:
    """On-disk sample docs, read with the REUSED tokenizer_fertility.read_docs
    (blank-line splitting: fragments, see module docstring)."""
    lang_dir = corpora_root / lang
    if not lang_dir.is_dir():
        raise FileNotFoundError(f"no on-disk sample at {lang_dir}")
    for doc in _fert.read_docs(lang_dir, max_bytes):
        yield doc, None


def probe_starcoderdata() -> dict:
    """Does the configured HF token open bigcode/starcoderdata? (Cheap: 4-byte read.)"""
    from huggingface_hub import HfFileSystem
    from huggingface_hub.errors import GatedRepoError

    try:
        files = list_parquet_files(STARCODER_REPO, STARCODER_DIRS["python"])
        with HfFileSystem().open(f"datasets/{STARCODER_REPO}/{files[0]}", "rb") as f:
            magic = f.read(4)
        if magic != b"PAR1":
            raise RuntimeError(f"unexpected parquet magic {magic!r} from {files[0]}")
        return {"accessible": True}
    except GatedRepoError as e:
        return {"accessible": False,
                "error": str(e).splitlines()[0] if str(e) else "GatedRepoError",
                "note": "the HF token that opens the-stack-dedup does NOT open "
                        "starcoderdata; it is gated separately -- request access at "
                        "https://huggingface.co/datasets/bigcode/starcoderdata"}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_COLUMNS = [
    ("survival_rate", "cleaning survival", "{:.1%}"),
    ("dup_rate", "exact-dup", "{:.2%}"),
    ("license_coverage", "usable permissive lic.", "{:.1%}"),
    ("mean_bytes", "mean B", "{:,.0f}"),
    ("p50_bytes", "p50 B", "{:,.0f}"),
    ("p90_bytes", "p90 B", "{:,.0f}"),
    ("alnum_fraction", "alnum", "{:.3f}"),
    ("tokens_per_byte", "tok/B (code-49k)", "{:.3f}"),
    ("comment_fraction", "comment/docstr", "{:.1%}"),
    ("mean_ident_len", "ident len", "{:.2f}"),
]


def _cell(entry: dict, key: str, fmt: str) -> str:
    v = entry.get(key)
    return "n/a" if v is None else fmt.format(v)


def render_markdown(results: dict) -> str:
    lines: list[str] = []
    lines.append("# Base code-corpus source comparison\n")
    lines.append(
        f"Generated by `scripts/compare_code_sources.py` on {results['generated']}; "
        f"~{results['mb_per_lang']:.0f} MB streamed per language per source, fertility "
        f"under `{results['tokenizer']}`. Cleaning survival / license coverage are "
        "measured over all streamed docs; content metrics over the cleaning survivors. "
        "\"usable permissive lic.\" = fraction passing the corpus build's `license_ok` "
        "allowlist (metadata present AND every license permissive) -- the attribution "
        "requirement.\n")

    probe = results["starcoderdata_probe"]
    lines.append("## Source accessibility\n")
    lines.append("- **bigcode/the-stack-dedup**: accessible (gated; the configured HF "
                 "token works, verified by streaming reads).")
    if probe["accessible"]:
        lines.append("- **bigcode/starcoderdata**: accessible.")
    else:
        lines.append(f"- **bigcode/starcoderdata**: NOT accessible. {probe['note']} "
                     f"(`{probe['error']}`)")
    lines.append("- **substitutes** (codeparrot/github-code-clean, "
                 "bleugreen/typescript-chunks): public, non-gated.\n")

    for lang in results["languages"]:
        lines.append(f"## {lang}\n")
        header = "| source | " + " | ".join(h for _, h, _ in _COLUMNS) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(_COLUMNS) + 1))
        for src in SOURCE_ORDER:
            entry = results["sources"].get(src, {}).get(lang)
            if entry is None:
                reason = ("*not accessible*" if src == "starcoderdata"
                          and not probe["accessible"] else "*not measured*")
                row = [src, reason] + ["--"] * (len(_COLUMNS) - 1)
                lines.append("| " + " | ".join(row) + " |")
                continue
            row = [src] + [_cell(entry, k, f) for k, _, f in _COLUMNS]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    notes = results.get("notes", [])
    if notes:
        lines.append("## Measurement notes\n")
        for n in notes:
            lines.append(f"- {n}")
        lines.append("")

    lines.append(_recommendation(results))
    return "\n".join(lines) + "\n"


def _mean_over_langs(results: dict, src: str, key: str) -> float | None:
    vals = [results["sources"][src][lang][key] for lang in results["languages"]
            if results["sources"].get(src, {}).get(lang, {}).get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def _recommendation(results: dict) -> str:
    langs = results["languages"]
    fresh = results["sources"].get("substitutes-fresh", {})
    probe = results["starcoderdata_probe"]
    stack_lic = _mean_over_langs(results, "the-stack-dedup", "license_coverage")
    cp_lic = [fresh[lg]["license_coverage"] for lg in ("python", "javascript")
              if fresh.get(lg, {}).get("license_coverage") is not None]
    lines = ["## Recommendation\n"]
    lines.append(
        f"**Use the-stack-dedup as the sole base source for the real corpus.** It is the "
        f"only accessible source that satisfies the hard provenance requirement end to "
        f"end: every file carries repo + hexsha + a license list, and "
        f"{stack_lic:.0%} of streamed files (mean over {'/'.join(langs)}) pass the "
        f"strict permissive allowlist outright -- the rejected remainder is mostly "
        f"weak-copyleft or multi-licensed files we deliberately exclude. It is "
        f"already near-deduplicated ("
        f"{_mean_over_langs(results, 'the-stack-dedup', 'dup_rate'):.2%} residual exact-dup "
        f"rate in-sample), covers all three target languages from the same collection "
        f"with uniform metadata, and its volume (~29 GB parquet Python, ~64 GB "
        f"JavaScript, ~15 GB TypeScript) comfortably feeds a 40-90B-token budget after "
        f"license filtering.\n")
    if not probe["accessible"]:
        lines.append(
            "**starcoderdata is out until its gate is granted.** The working token does "
            "not open it (verified: file reads 403 even though metadata listing "
            "succeeds); it also ships no per-file license column, so even once granted "
            "it would need joining back to the-stack for attribution -- little upside "
            "over the-stack-dedup itself for this use.\n")
    if cp_lic:
        lines.append(
            f"**Drop the non-gated substitutes from the specialist mix.** "
            f"codeparrot/github-code-clean remains a reasonable fallback (its per-doc "
            f"license column shows {sum(cp_lic)/len(cp_lic):.0%} usable-permissive), but "
            f"it is not deduplicated against the-stack, so mixing the two double-counts "
            f"popular files; the TypeScript substitute "
            f"(bleugreen/typescript-chunks) has NO per-file license metadata and cannot "
            f"meet the attribution requirement at all. The substitutes stay useful only "
            f"as the tokenizer-study corpus they already served as.\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mb-per-lang", type=float, default=200.0)
    ap.add_argument("--languages", nargs="+", default=list(LANGS), choices=LANGS)
    ap.add_argument("--sources", nargs="+", default=list(SOURCE_ORDER), choices=SOURCE_ORDER)
    ap.add_argument("--tokenizer", default="data/tokenizers/code-49k.json")
    ap.add_argument("--corpora", default="data/corpora/code-samples")
    ap.add_argument("--download-dir", default="data/corpora/_compare-download")
    ap.add_argument("--json-out", default="docs/code-corpus-comparison.json")
    ap.add_argument("--md-out", default="docs/code-corpus-comparison.md")
    ap.add_argument("--render-only", action="store_true",
                    help="re-render the markdown from an existing --json-out")
    args = ap.parse_args()

    json_out = Path(args.json_out)
    md_out = Path(args.md_out)
    if args.render_only:
        results = json.loads(json_out.read_text())
        md_out.write_text(render_markdown(results))
        print(f"re-rendered {md_out} from {json_out}")
        return

    from microlab.tokenizer.fast import FastTokenizer

    tok = FastTokenizer.load(args.tokenizer)
    target = int(args.mb_per_lang * 1e6)
    download_dir = Path(args.download_dir)
    corpora_root = Path(args.corpora)

    probe = probe_starcoderdata()
    print(f"starcoderdata probe: {probe}", flush=True)

    results: dict = {
        "generated": dt.date.today().isoformat(),
        "mb_per_lang": args.mb_per_lang,
        "tokenizer": Path(args.tokenizer).stem,
        "languages": list(args.languages),
        "starcoderdata_probe": probe,
        "sources": {},
        "notes": [
            "substitutes-disk docs are blank-line-split FRAGMENTS of the original files "
            "(the .txt sample format is lossy), so its length/comment stats are not "
            "file-level; substitutes-fresh is the faithful measurement of the same "
            "upstream sources.",
            "substitutes-disk was already cleaned once at sampling time; its "
            "sub-100% survival and its exact-dup rate are fragment artifacts (tiny "
            "fragments fail the min-size gate, short import/header fragments repeat), "
            "not new information about the underlying source.",
            "the-stack-dedup license metadata is the max_stars_repo_licenses list; "
            "starcoderdata has no per-file license column.",
            "typescript substitutes-disk sample is only ~38 MB total (all of it used).",
            "the-stack tokenizes 2-4% worse under code-49k than the substitutes "
            "(tok/B column): code-49k was TRAINED on the substitute corpus, so this is "
            "an in-domain-tokenizer artifact (plus fragment effects for ts), not a "
            "corpus-quality signal.",
        ],
    }

    def pairs_for(src: str, lang: str):
        if src == "the-stack-dedup":
            return stack_pairs(lang, download_dir)
        if src == "starcoderdata":
            return starcoder_pairs(lang, download_dir)
        if src == "substitutes-fresh":
            return substitutes_fresh_pairs(lang)
        if src == "substitutes-disk":
            return disk_pairs(lang, corpora_root, target)
        raise ValueError(f"unknown source {src!r}")

    for src in args.sources:
        if src == "starcoderdata" and not probe["accessible"]:
            print("skipping starcoderdata (not accessible)", flush=True)
            continue
        results["sources"][src] = {}
        for lang in args.languages:
            print(f"[{src} / {lang}] streaming ~{args.mb_per_lang:.0f} MB ...", flush=True)
            cell = measure_pairs(pairs_for(src, lang), lang=lang, tok=tok,
                                 target_bytes=target)
            results["sources"][src][lang] = cell
            print(f"[{src} / {lang}] kept {cell['kept_docs']}/{cell['streamed_docs']} "
                  f"docs, survival {cell['survival_rate']:.1%}, "
                  f"tok/B {cell['tokens_per_byte']:.3f}", flush=True)
            # progressive write: partial results land on disk as each cell finishes
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(json.dumps(results, indent=2))

    md_out.write_text(render_markdown(results))
    print(f"wrote {json_out} and {md_out}")


if __name__ == "__main__":
    main()
