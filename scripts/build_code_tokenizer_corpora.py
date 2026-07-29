"""Sample representative code + prose corpora for the code-native tokenizer study.

Streams permissive-licensed source datasets from the HuggingFace Hub (no auth: every
source below is public and non-gated), applies light cleaning (drop non-UTF-8, drop
minified/generated blobs), and writes raw `.txt` samples under
`data/corpora/code-samples/<lang>/` plus a `manifest.json` recording counts, source
ids, license provenance, and any substitution notes.

    python scripts/build_code_tokenizer_corpora.py            # full sample (~1.5 GB text)
    python scripts/build_code_tokenizer_corpora.py --smoke    # tiny sample for a dry run

Source ladder (chosen for streamable, non-gated, permissive availability; the-stack-v2
is gated and needs Software-Heritage resolution, so per-language substitutes are used and
NOTED in the manifest):

  python     codeparrot/github-code-clean (parquet branch, Python-all)     real repos, license col
  javascript codeparrot/github-code-clean (parquet branch, JavaScript-all) real repos, license col
  typescript bleugreen/typescript-chunks                    SUBSTITUTE for gated stack-v2 TS
  shell      ajibawa-2023/Shell-Code-Large                  SUBSTITUTE (curated shell)
  sql        gretelai/synthetic_text_to_sql                 SUBSTITUTE (synthetic SQL syntax)
  json       ibragim-bad/github-repos-metadata-40M          SUBSTITUTE (real metadata -> json.dumps)
  markdown   open-index/open-markdown-v2                     real crawled markdown
  prose      local FineWeb shards decoded w/ baseline tok    reuses data/shards/fineweb

Cleaning heuristics (thresholds documented on `looks_minified`): a document is dropped as
minified/generated when its longest line is very long or the bulk of its bytes live in very
long lines -- the signature of bundled/minified JS, single-line generated blobs, etc.
Non-UTF-8 documents (lone surrogates, undecodable bytes) are dropped too. Prose/markdown use a
laxer line-length ceiling because natural long paragraphs are legitimate there.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Cleaning heuristics (pure; unit-tested)
# ---------------------------------------------------------------------------

# A line at/above this length is essentially never hand-written source; it is the
# hallmark of minified/bundled code or a generated single-line blob.
MINIFIED_MAX_LINE = 1000
# Hard ceiling for prose/markdown. Natural text is frequently unwrapped (a whole
# paragraph on one line), so the code line-length rules would wrongly flag it; only a
# genuinely pathological single line (base64/HTML dump, generated blob) trips this.
PROSE_MAX_LINE = 5000
# If most of a document's bytes live in "long" (>= this) lines, code reads as machine
# generated even when no single line hits the hard ceiling. NOT applied to prose, and
# only to documents >= LONG_LINE_MIN_TOTAL: minified bundles are large files, whereas a
# short single-statement doc (one long SQL query) legitimately has one long line.
LONG_LINE_CHARS = 500
LONG_LINE_BYTE_FRAC = 0.5
LONG_LINE_MIN_TOTAL = 2000


def is_utf8(text: str) -> bool:
    """True when `text` is clean UTF-8 (no lone surrogates / undecodable content)."""
    try:
        text.encode("utf-8", "strict")
    except UnicodeEncodeError:
        return False
    return True


def looks_minified(text: str, *, code: bool = True) -> bool:
    """Heuristic minified/generated-blob detector.

    For CODE, flag a document when either its longest line reaches `MINIFIED_MAX_LINE`
    (the classic single-line minified-bundle signature) or at least `LONG_LINE_BYTE_FRAC`
    of its bytes sit in lines >= `LONG_LINE_CHARS` (dense generated code that keeps a line
    cap) -- the latter only for docs >= `LONG_LINE_MIN_TOTAL` so short single-statement
    files (one long SQL query) are not misflagged. For PROSE/markdown, apply only the
    much higher `PROSE_MAX_LINE` hard ceiling:
    unwrapped natural-text paragraphs are legitimately one long line, so the code rules
    would misfire, but a 5k+ char single line is still a generated/pathological blob.
    """
    if not text:
        return True
    lines = text.split("\n")
    max_line = max((len(line) for line in lines), default=0)
    if not code:
        return max_line >= PROSE_MAX_LINE
    if max_line >= MINIFIED_MAX_LINE:
        return True
    total = len(text)
    if total < LONG_LINE_MIN_TOTAL:
        return False
    long_bytes = sum(len(line) for line in lines if len(line) >= LONG_LINE_CHARS)
    return long_bytes / total >= LONG_LINE_BYTE_FRAC


def clean_documents(
    docs: Iterator[str],
    *,
    code: bool,
    min_chars: int,
    max_chars: int,
) -> Iterator[tuple[str, str | None]]:
    """Yield `(text, drop_reason)`; `drop_reason is None` for kept docs.

    Docs shorter than `min_chars` or longer than `max_chars` are skipped (too trivial /
    a single huge blob that would swamp diversity) but not counted as minified/non-utf8.
    """
    for raw in docs:
        if raw is None:
            continue
        text = raw if raw.endswith("\n") else raw + "\n"
        if len(text) < min_chars or len(text) > max_chars:
            yield text, "size"
            continue
        if not is_utf8(text):
            yield text, "nonutf8"
            continue
        if looks_minified(text, code=code):
            yield text, "minified"
            continue
        yield text, None


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------


@dataclass
class Source:
    lang: str
    code: bool
    source_id: str
    license_note: str
    substitution: str | None = None
    # loader-specific config resolved in `iter_source`
    hf_kwargs: dict = field(default_factory=dict)
    text_fields: tuple[str, ...] = ("content",)


SOURCES: dict[str, Source] = {
    "python": Source(
        lang="python", code=True, source_id="codeparrot/github-code-clean",
        license_note="per-doc `license` column (permissive OSS licenses)",
        hf_kwargs=dict(data_files="Python-all/partial-train/*.parquet",
                       revision="refs/convert/parquet", split="train"),
        text_fields=("code",)),
    "javascript": Source(
        lang="javascript", code=True, source_id="codeparrot/github-code-clean",
        license_note="per-doc `license` column (permissive OSS licenses)",
        hf_kwargs=dict(data_files="JavaScript-all/partial-train/*.parquet",
                       revision="refs/convert/parquet", split="train"),
        text_fields=("code",)),
    "typescript": Source(
        lang="typescript", code=True, source_id="bleugreen/typescript-chunks",
        license_note="upstream repo licenses (public dataset; licenses not per-doc labelled)",
        substitution="the-stack-v2 TypeScript is gated (needs SWH resolution); "
                     "bleugreen/typescript-chunks is a streamable non-gated TS substitute",
        hf_kwargs=dict(split="train"), text_fields=("content",)),
    "shell": Source(
        lang="shell", code=True, source_id="ajibawa-2023/Shell-Code-Large",
        license_note="public dataset of shell scripts (licenses not per-doc labelled)",
        substitution="no non-gated the-stack Shell config; curated shell-code dataset substitute",
        hf_kwargs=dict(split="train"), text_fields=("code",)),
    "sql": Source(
        lang="sql", code=True, source_id="gretelai/synthetic_text_to_sql",
        license_note="Apache-2.0 (Gretel synthetic)",
        substitution="no non-gated the-stack SQL config; synthetic-SQL dataset substitute "
                     "(real SQL syntax: DDL in sql_context + query in sql)",
        hf_kwargs=dict(split="train"), text_fields=("sql_context", "sql")),
    "json": Source(
        lang="json", code=True, source_id="ibragim-bad/github-repos-metadata-40M",
        license_note="public GitHub metadata (facts; serialized to JSON here)",
        substitution="no non-gated raw-JSON-file corpus; real GitHub repo metadata rows "
                     "serialized with json.dumps(indent=2) to produce representative JSON syntax",
        hf_kwargs=dict(split="sample"), text_fields=("__json__",)),
    "markdown": Source(
        lang="markdown", code=False, source_id="open-index/open-markdown-v2",
        license_note="crawled markdown documents (source-site licenses vary)",
        hf_kwargs=dict(split="train"), text_fields=("markdown",)),
}


def iter_source(src: Source) -> Iterator[str]:
    """Stream raw text documents from a HuggingFace source, one string per document."""
    from datasets import load_dataset  # heavy/optional dep, imported lazily

    ds = load_dataset(src.source_id, streaming=True, **src.hf_kwargs)
    for row in ds:
        if src.text_fields == ("__json__",):
            # Serialize a real metadata row to pretty JSON (representative JSON syntax).
            yield json.dumps(row, indent=2, ensure_ascii=False, default=str)
            continue
        parts = [str(row[f]) for f in src.text_fields if row.get(f)]
        if parts:
            yield "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Prose (local FineWeb shards, decoded offline with the baseline tokenizer)
# ---------------------------------------------------------------------------


def iter_prose(shard_dir: Path, tokenizer_path: Path, max_tokens: int) -> Iterator[str]:
    """Decode locally-tokenized FineWeb shards back to prose documents (offline).

    Reads uint16 token shards, splits on the EOT token into documents, and decodes each
    with the baseline tokenizer -- reusing the exact FineWeb prose the model trains on.
    """
    import numpy as np

    from microlab.tokenizer.fast import FastTokenizer

    tok = FastTokenizer.load(str(tokenizer_path))
    eot = tok.eot_token
    manifest = json.loads((shard_dir / "train-manifest.json").read_text())
    consumed = 0
    buf: list[int] = []
    for shard in manifest["shards"]:
        arr = np.fromfile(shard_dir / shard["file"], dtype=np.uint16)
        for tid in arr.tolist():
            if tid == eot:
                if buf:
                    yield tok.decode(buf)
                    buf = []
            else:
                buf.append(tid)
            consumed += 1
            if consumed >= max_tokens:
                if buf:
                    yield tok.decode(buf)
                return


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def sample_language(
    docs: Iterator[str],
    out_dir: Path,
    *,
    code: bool,
    target_bytes: int,
    min_chars: int,
    max_chars: int,
    docs_per_file: int = 2000,
) -> dict:
    """Write cleaned docs to sharded `.txt` files until `target_bytes` of kept text.

    Documents are separated by a blank line; each `.txt` file holds up to `docs_per_file`
    documents. Returns manifest stats for this language.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for existing in out_dir.glob("*.txt"):
        existing.unlink()

    kept = kept_bytes = kept_lines = 0
    dropped = {"minified": 0, "nonutf8": 0, "size": 0}
    file_idx = 0
    handle = None
    in_file = 0

    def _open(idx: int):
        return (out_dir / f"sample-{idx:04d}.txt").open("w", encoding="utf-8")

    for text, reason in clean_documents(docs, code=code, min_chars=min_chars,
                                        max_chars=max_chars):
        if reason is not None:
            if reason in dropped:
                dropped[reason] += 1
            continue
        if handle is None or in_file >= docs_per_file:
            if handle is not None:
                handle.close()
                file_idx += 1
            handle = _open(file_idx)
            in_file = 0
        handle.write(text)
        handle.write("\n")
        in_file += 1
        kept += 1
        kept_bytes += len(text.encode("utf-8"))
        kept_lines += text.count("\n")
        if kept_bytes >= target_bytes:
            break
    if handle is not None:
        handle.close()

    return {
        "documents": kept,
        "bytes": kept_bytes,
        "lines": kept_lines,
        "files": file_idx + 1 if kept else 0,
        "dropped_minified": dropped["minified"],
        "dropped_nonutf8": dropped["nonutf8"],
        "dropped_size": dropped["size"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/corpora/code-samples")
    ap.add_argument("--shard-dir", default="data/shards/fineweb",
                    help="local FineWeb shards to decode for the prose sample")
    ap.add_argument("--min-chars", type=int, default=64)
    ap.add_argument("--max-chars", type=int, default=1_000_000,
                    help="skip single docs bigger than this (avoid one blob swamping the mix)")
    ap.add_argument("--smoke", action="store_true", help="tiny sample for a dry run")
    ap.add_argument("--only", nargs="*", help="restrict to these langs (default: all)")
    args = ap.parse_args()

    # Target raw-text bytes per language (primary code langs get the largest samples).
    mb = 1_000_000
    if args.smoke:
        targets = {k: 1 * mb for k in SOURCES} | {"prose": 1 * mb}
        prose_tokens = 300_000
    else:
        targets = {
            "python": 300 * mb, "javascript": 300 * mb, "typescript": 300 * mb,
            "shell": 100 * mb, "sql": 100 * mb, "json": 100 * mb, "markdown": 100 * mb,
            "prose": 200 * mb,
        }
        prose_tokens = 60_000_000

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    langs = args.only or [*SOURCES.keys(), "prose"]

    manifest: dict = {"source_root": str(out_root), "languages": {}}

    for lang in langs:
        target = targets[lang]
        print(f"[{lang}] sampling ~{target/mb:.0f} MB ...", flush=True)
        if lang == "prose":
            docs = iter_prose(Path(args.shard_dir),
                              Path(args.shard_dir) / "tokenizer.json", prose_tokens)
            stats = sample_language(docs, out_root / lang, code=False, target_bytes=target,
                                    min_chars=args.min_chars, max_chars=args.max_chars)
            stats |= {"source": f"local FineWeb shards ({args.shard_dir})",
                      "license": "FineWeb (ODC-By); reused local pretraining shards",
                      "substitution": None, "code": False}
        else:
            src = SOURCES[lang]
            stats = sample_language(iter_source(src), out_root / lang, code=src.code,
                                    target_bytes=target, min_chars=args.min_chars,
                                    max_chars=args.max_chars)
            stats |= {"source": src.source_id, "license": src.license_note,
                      "substitution": src.substitution, "code": src.code}
        manifest["languages"][lang] = stats
        print(f"[{lang}] kept {stats['documents']} docs / {stats['bytes']/mb:.1f} MB "
              f"(dropped: minified={stats['dropped_minified']} "
              f"nonutf8={stats['dropped_nonutf8']} size={stats['dropped_size']})", flush=True)

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote manifest -> {out_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
