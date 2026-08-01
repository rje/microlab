"""Build the coding-specialist pretraining corpus from bigcode/the-stack-dedup (gated).

Pipeline (every stage streams; nothing buffers the whole corpus):

  per-language parquet files (hf_hub_download one at a time, deleted after use)
    -> language check (partition dir + `lang` column; a mismatch raises)
    -> cleaning gates REUSED from scripts/build_code_tokenizer_corpora.py
       (non-UTF-8, minified/generated blobs, size bounds -- single source of truth)
    -> license filter: permissive allowlist over the per-file license metadata, plus an
       attribution-completeness gate (repo + hexsha + path required); every kept file
       appends {lang, repo, hexsha, path, licenses, split, tokens} to attribution.jsonl
       -- the shipped attribution manifest (hard requirement)
    -> exact dedup on sha256(content) (near-dedup: see the `near_dup_reason` hook)
    -> deterministic train/val routing by content hash (--val-fraction)
    -> tokenize with --tokenizer (EOT appended after each document)
    -> uint16 .bin shards in EXACTLY the ShardDataset layout of data/shards/fineweb-100bt:
       <split>-NNNNN.bin + <split>-manifest.json, tokenizer.json copied alongside.

Resumable + progressive (house rule: long jobs write as they go). A checkpoint is taken
every --checkpoint-rows source rows and at every language boundary; it persists the
partial token buffers (<split>-pending-<ckpt>.u16), appends buffered attribution records
and dedup hashes, then atomically replaces build-state.json (per-language source cursor =
(parquet file index, rows consumed), per-split shard counts, attribution/hash offsets).
Re-running with the same --out resumes: shard manifests are truncated back to the
checkpointed counts (orphan shards from a mid-write crash are deleted and regenerated),
attribution.jsonl / seen-hashes.u64 are truncated to the checkpointed offsets, and the
stream restarts at the cursor -- no duplicated and no lost records, whatever instant the
previous run died.

Near-dedup hook: only exact-hash dedup is implemented today. MinHash/simhash plugs into
`near_dup_reason` (documented there); `--near-dedup` deliberately raises until then.

Smoke build (~200M tokens):

    python scripts/build_code_corpus.py --out data/shards/code-stack-smoke \\
        --tokenizer data/tokenizers/code-49k.json \\
        --languages python javascript typescript \\
        --target-tokens 200_000_000 --val-fraction 0.005
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

SCRIPTS_DIR = Path(__file__).resolve().parent


def load_script_module(name: str):
    """Import a sibling scripts/ module by path (scripts/ is not a package)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register before exec so dataclasses resolve __module__
    spec.loader.exec_module(mod)
    return mod


# REUSED cleaning gates -- the same code that filtered the tokenizer-training corpus.
_corpora = load_script_module("build_code_tokenizer_corpora")


def clean_one(raw: str, *, code: bool = True, min_chars: int = 64,
              max_chars: int = 1_000_000) -> tuple[str, str | None]:
    """Run ONE document through the reused tokenizer-corpus cleaning gates.

    Returns `(text, drop_reason)`; reason is None for kept docs, else one of
    "size" / "nonutf8" / "minified" (see build_code_tokenizer_corpora.clean_documents).
    """
    if raw is None:
        raise ValueError("null document content (source contract violation)")
    return next(_corpora.clean_documents(
        iter([raw]), code=code, min_chars=min_chars, max_chars=max_chars))


# ---------------------------------------------------------------------------
# License filter (permissive allowlist) -- the clean-provenance gate
# ---------------------------------------------------------------------------

# Conservative SPDX-style allowlist (compared case-insensitively). Deliberately narrower
# than the-stack's own "permissive" umbrella: weak-copyleft (MPL/EPL/LGPL), Artistic,
# CDDL etc. are excluded, as is anything unknown.
PERMISSIVE_LICENSES = frozenset({
    "mit", "mit-0", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "bsd-3-clause-clear",
    "isc", "unlicense", "0bsd", "zlib", "cc0-1.0",
})


def license_ok(licenses) -> bool:
    """True iff there IS license metadata and EVERY listed license is on the allowlist.

    A file tagged e.g. ["MIT", "GPL-2.0"] is rejected: the copyleft grant still binds.
    Missing/empty metadata is rejected too -- unknown provenance cannot be attributed.
    """
    if not licenses:
        return False
    return all(str(lic).strip().lower() in PERMISSIVE_LICENSES for lic in licenses)


# ---------------------------------------------------------------------------
# Source: bigcode/the-stack-dedup parquet streaming with a precise resume cursor
# ---------------------------------------------------------------------------

STACK_REPO = "bigcode/the-stack-dedup"
STACK_DIRS = {"python": "data/python", "javascript": "data/javascript",
              "typescript": "data/typescript", "markdown": "data/markdown"}
# Markdown is prose, not code, and it is in the mix for a different reason: READMEs and
# docs are where a code model sees natural-language explanation of code. It flows through
# the identical license-filter and attribution path as the source languages, which is the
# point of adding it here rather than via the plain text ingest.
STACK_LANG_NAMES = {"python": "Python", "javascript": "JavaScript",
                    "typescript": "TypeScript", "markdown": "Markdown"}
PROSE_LANGS = frozenset({"markdown"})
STACK_COLUMNS = ["content", "hexsha", "max_stars_repo_name", "max_stars_repo_path",
                 "max_stars_repo_licenses", "lang"]


@dataclass
class SourceRow:
    """One source file plus its attribution metadata and stream position."""
    content: str
    hexsha: str | None
    repo: str | None
    path: str | None
    licenses: list[str]
    lang: str | None
    file_idx: int  # index into the language's sorted parquet-file list
    row_idx: int   # absolute row index within that parquet file


def list_parquet_files(repo: str, subdir: str) -> list[str]:
    """Sorted parquet paths (relative to the dataset repo root) under `subdir`."""
    from huggingface_hub import HfFileSystem

    prefix = f"datasets/{repo}/"
    names = sorted(n for n in HfFileSystem().ls(f"{prefix}{subdir}", detail=False)
                   if n.endswith(".parquet"))
    if not names:
        raise FileNotFoundError(f"no parquet files under {prefix}{subdir}")
    return [n.removeprefix(prefix) for n in names]


def groups_after_skip(group_rows: list[int], skip: int) -> tuple[list[int], int]:
    """Which parquet row groups to read, and how many rows to discard from the first
    one, so that reading resumes at absolute row `skip`. ([], 0) when `skip` covers the
    whole file. Pure cursor math; unit-tested."""
    if skip < 0:
        raise ValueError(f"negative skip {skip}")
    off = 0
    for gi, n in enumerate(group_rows):
        if off + n > skip:
            return list(range(gi, len(group_rows))), skip - off
        off += n
    return [], 0


def iter_stack_rows(lang: str, *, download_dir: str | Path, start_file: int = 0,
                    start_row: int = 0, batch_rows: int = 1024,
                    repo: str = STACK_REPO) -> Iterator[SourceRow]:
    """Stream SourceRows for `lang` in deterministic order, resumable at
    (start_file, start_row).

    Each parquet file is downloaded whole (hf_hub_download saturates the link at
    ~40 MB/s vs <10 MB/s for remote fsspec reads), parsed locally, then deleted --
    transient disk use stays at one file (~200-500 MB).
    """
    import pyarrow.parquet as papq
    from huggingface_hub import hf_hub_download

    files = list_parquet_files(repo, STACK_DIRS[lang])
    dl = Path(download_dir)
    dl.mkdir(parents=True, exist_ok=True)
    for fi in range(start_file, len(files)):
        local = hf_hub_download(repo, files[fi], repo_type="dataset", local_dir=str(dl))
        try:
            pf = papq.ParquetFile(local)
            n_groups = pf.metadata.num_row_groups
            group_rows = [pf.metadata.row_group(g).num_rows for g in range(n_groups)]
            skip = start_row if fi == start_file else 0
            read_groups, drop = groups_after_skip(group_rows, skip)
            if not read_groups:
                continue
            abs_idx = skip - drop  # absolute index of the first row actually read
            for batch in pf.iter_batches(batch_size=batch_rows, row_groups=read_groups,
                                         columns=STACK_COLUMNS):
                cols = {c: batch.column(c) for c in batch.schema.names}
                for i in range(batch.num_rows):
                    if drop:
                        drop -= 1
                        abs_idx += 1
                        continue
                    yield SourceRow(
                        content=cols["content"][i].as_py(),
                        hexsha=cols["hexsha"][i].as_py(),
                        repo=cols["max_stars_repo_name"][i].as_py(),
                        path=cols["max_stars_repo_path"][i].as_py(),
                        licenses=cols["max_stars_repo_licenses"][i].as_py() or [],
                        lang=cols["lang"][i].as_py(),
                        file_idx=fi, row_idx=abs_idx)
                    abs_idx += 1
        finally:
            Path(local).unlink(missing_ok=False)


# ---------------------------------------------------------------------------
# Near-dedup hook
# ---------------------------------------------------------------------------


def near_dup_reason(text: str) -> str | None:
    """NEAR-DEDUP HOOK -- not implemented yet; exact-hash dedup only.

    This is the seam where MinHash/simhash near-duplicate detection plugs in: return a
    drop-reason string (e.g. "near_dup") to drop `text`, None to keep it. An
    implementation must stay deterministic and resume-safe (persist its state via an
    AppendLog flushed in CorpusBuilder._checkpoint, like seen-hashes.u64). Until then
    this keeps everything, and the `--near-dedup` CLI flag raises rather than silently
    pretending coverage.
    """
    return None


# ---------------------------------------------------------------------------
# Resumable writers
# ---------------------------------------------------------------------------


class ShardWriter:
    """Resumable uint16 shard writer emitting EXACTLY the ShardDataset layout
    (<split>-NNNNN.bin + <split>-manifest.json, as in data/shards/fineweb-100bt).

    Progressive: the manifest is atomically rewritten after every completed shard, so a
    valid manifest always exists. `checkpoint(k)` persists the partial in-memory buffer
    to <split>-pending-<k>.u16; `resume(n_shards, k)` truncates the shard list back to
    the checkpointed count (deleting newer orphan shards written after the last
    checkpoint -- their documents will be reprocessed) and reloads the matching pending
    buffer, so writer state exactly matches the checkpoint.
    """

    def __init__(self, out_dir: Path, split: str, shard_size: int) -> None:
        self.out = out_dir
        self.split = split
        self.shard_size = shard_size
        self.shards: list[dict] = []
        self.pending: list[np.ndarray] = []
        self.pending_len = 0

    @property
    def total_tokens(self) -> int:
        return sum(s["tokens"] for s in self.shards)

    @property
    def manifest_path(self) -> Path:
        return self.out / f"{self.split}-manifest.json"

    def _write_manifest(self) -> None:
        manifest = {"split": self.split, "shards": self.shards,
                    "total_tokens": self.total_tokens, "dtype": "uint16"}
        tmp = self.out / f"{self.split}-manifest.json.tmp"
        tmp.write_text(json.dumps(manifest, indent=2))
        os.replace(tmp, self.manifest_path)

    def _write_shard(self, data: np.ndarray) -> None:
        name = f"{self.split}-{len(self.shards):05d}.bin"
        data.astype(np.uint16, copy=False).tofile(self.out / name)
        self.shards.append({"file": name, "tokens": int(len(data))})
        self._write_manifest()

    def append(self, arr: np.ndarray) -> None:
        arr = np.asarray(arr, dtype=np.uint16).reshape(-1)
        if not len(arr):
            return
        self.pending.append(arr)
        self.pending_len += len(arr)
        while self.pending_len >= self.shard_size:
            data = np.concatenate(self.pending) if len(self.pending) > 1 else self.pending[0]
            self._write_shard(data[:self.shard_size])
            rest = data[self.shard_size:]
            self.pending = [rest] if len(rest) else []
            self.pending_len = int(len(rest))

    def pending_file(self, ckpt: int) -> Path:
        return self.out / f"{self.split}-pending-{ckpt:06d}.u16"

    def checkpoint(self, ckpt: int) -> None:
        buf = (np.concatenate(self.pending) if len(self.pending) > 1
               else self.pending[0] if self.pending else np.empty(0, dtype=np.uint16))
        buf.astype(np.uint16, copy=False).tofile(self.pending_file(ckpt))

    def drop_pending_files(self, keep_ckpt: int | None) -> None:
        for p in self.out.glob(f"{self.split}-pending-*.u16"):
            if keep_ckpt is None or p != self.pending_file(keep_ckpt):
                p.unlink()

    def resume(self, n_shards: int, ckpt: int) -> None:
        on_disk = (json.loads(self.manifest_path.read_text())["shards"]
                   if self.manifest_path.exists() else [])
        if len(on_disk) < n_shards:
            raise RuntimeError(
                f"{self.manifest_path} lists {len(on_disk)} shards but the checkpoint "
                f"recorded {n_shards}; corpus dir is corrupted")
        for orphan in on_disk[n_shards:]:  # written after the last checkpoint: regenerate
            (self.out / orphan["file"]).unlink(missing_ok=True)
        self.shards = on_disk[:n_shards]
        for s in self.shards:
            f = self.out / s["file"]
            if not f.exists() or f.stat().st_size != s["tokens"] * 2:
                raise RuntimeError(f"shard {f} missing or size-mismatched; "
                                   "corpus dir is corrupted")
        self._write_manifest()
        pf = self.pending_file(ckpt)
        if not pf.exists():
            raise RuntimeError(f"missing pending buffer {pf} for checkpoint {ckpt}")
        buf = np.fromfile(pf, dtype=np.uint16)
        self.pending = [buf] if len(buf) else []
        self.pending_len = int(len(buf))
        self.drop_pending_files(keep_ckpt=ckpt)

    def finalize(self) -> None:
        if self.pending_len:
            data = np.concatenate(self.pending) if len(self.pending) > 1 else self.pending[0]
            self._write_shard(data)
            self.pending, self.pending_len = [], 0
        self._write_manifest()
        self.drop_pending_files(keep_ckpt=None)


class AppendLog:
    """Append-only file with truncate-to-offset resume.

    Records buffer in memory and hit disk only at flush() (called from checkpoints). On
    resume the file is truncated back to the last checkpointed offset, so records
    written after the checkpoint -- whose documents get reprocessed -- are not
    duplicated.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.buffer: list[bytes] = []

    def add(self, payload: bytes) -> None:
        self.buffer.append(payload)

    def flush(self) -> int:
        """Append buffered records; returns the resulting file size in bytes."""
        if self.buffer:
            with self.path.open("ab") as f:
                f.write(b"".join(self.buffer))
            self.buffer = []
        return self.path.stat().st_size if self.path.exists() else 0

    def truncate(self, offset: int) -> None:
        if offset == 0:
            self.path.unlink(missing_ok=True)
            return
        size = self.path.stat().st_size
        if size < offset:
            raise RuntimeError(f"{self.path} is {size} bytes, shorter than the "
                               f"checkpointed offset {offset}; corpus dir is corrupted")
        with self.path.open("r+b") as f:
            f.truncate(offset)


class HashStore:
    """Exact-dedup store: 8-byte sha256 prefixes in memory, persisted append-only to
    seen-hashes.u64 (truncated to the checkpointed count on resume)."""

    def __init__(self, path: Path) -> None:
        self.log = AppendLog(path)
        self.seen: set[bytes] = set()

    def load(self, count: int) -> None:
        self.log.truncate(count * 8)
        if count:
            data = self.log.path.read_bytes()
            self.seen = {data[i:i + 8] for i in range(0, len(data), 8)}

    def add(self, digest8: bytes) -> bool:
        """Record `digest8`; True if it was new (keep the doc), False if seen (dup)."""
        if digest8 in self.seen:
            return False
        self.seen.add(digest8)
        self.log.add(digest8)
        return True

    def flush(self) -> int:
        self.log.flush()
        return len(self.seen)


# ---------------------------------------------------------------------------
# Build driver
# ---------------------------------------------------------------------------


@dataclass
class BuildConfig:
    out: Path
    tokenizer: str
    languages: list[str]
    target_tokens: int
    val_fraction: float
    weights: list[float]
    shard_size: int = 100_000_000
    batch_docs: int = 512
    checkpoint_rows: int = 50_000
    min_chars: int = 64
    max_chars: int = 1_000_000
    source_repo: str = STACK_REPO

    def identity(self) -> dict:
        """The stream-defining knobs. A resume with different values would silently
        corrupt the corpus, so build-state.json pins them and a mismatch raises.
        (batch_docs / checkpoint_rows only affect flush timing and MAY change.)"""
        return {"tokenizer": str(self.tokenizer), "languages": list(self.languages),
                "target_tokens": self.target_tokens, "val_fraction": self.val_fraction,
                "weights": list(self.weights), "shard_size": self.shard_size,
                "min_chars": self.min_chars, "max_chars": self.max_chars,
                "source_repo": self.source_repo}


DROP_KEYS = ("dropped_size", "dropped_nonutf8", "dropped_minified", "dropped_license",
             "dropped_attribution", "dropped_dup", "dropped_near_dup")


def _new_lang_state() -> dict:
    return {"file_idx": 0, "rows_done": 0, "done": False, "exhausted": False, "rows": 0,
            "kept_docs": 0, "train_tokens": 0, "val_tokens": 0,
            **{k: 0 for k in DROP_KEYS}}


class CorpusBuilder:
    """Runs the pipeline against a rows factory (`iter_stack_rows` in production;
    injectable for tests). All state lives in --out; see the module docstring for the
    checkpoint/resume contract."""

    def __init__(self, cfg: BuildConfig, rows_factory=iter_stack_rows) -> None:
        from microlab.tokenizer.fast import FastTokenizer

        self.cfg = cfg
        self.rows_factory = rows_factory
        cfg.out.mkdir(parents=True, exist_ok=True)
        self.tok = FastTokenizer.load(str(cfg.tokenizer))
        if self.tok.vocab_size > 65536:
            raise ValueError(f"vocab {self.tok.vocab_size} does not fit uint16 shards")
        if self.tok.eot_token is None:
            raise ValueError(f"tokenizer {cfg.tokenizer} lacks the <|endoftext|> token")
        self.state_path = cfg.out / "build-state.json"
        self.attrib = AppendLog(cfg.out / "attribution.jsonl")
        self.hashes = HashStore(cfg.out / "seen-hashes.u64")
        self.writers = {"train": ShardWriter(cfg.out, "train", cfg.shard_size)}
        if cfg.val_fraction > 0:
            self.writers["val"] = ShardWriter(cfg.out, "val", cfg.shard_size)
        self.batches: dict[str, list[tuple[str, dict]]] = {s: [] for s in self.writers}
        self.state = self._load_or_init_state()
        self.session_t0 = time.time()

    # -- state ------------------------------------------------------------

    def _load_or_init_state(self) -> dict:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text())
            if state["config"] != self.cfg.identity():
                raise RuntimeError(
                    f"config mismatch vs {self.state_path}; refusing to resume with "
                    f"different stream-defining settings.\n  recorded: {state['config']}\n"
                    f"  requested: {self.cfg.identity()}")
            if not state["completed"]:
                for split, w in self.writers.items():
                    w.resume(state["writers"][split]["shards"], state["ckpt"])
                self.attrib.truncate(state["attribution_bytes"])
                self.hashes.load(state["hashes_count"])
            return state
        for name in ("train-manifest.json", "val-manifest.json", "attribution.jsonl"):
            if (self.cfg.out / name).exists():
                raise FileExistsError(
                    f"{self.cfg.out / name} exists but there is no build-state.json; "
                    "refusing to overwrite a foreign corpus dir (or one whose build "
                    "crashed before its first checkpoint -- delete it to start over)")
        shutil.copyfile(self.cfg.tokenizer, self.cfg.out / "tokenizer.json")
        return {"config": self.cfg.identity(), "ckpt": 0, "completed": False,
                "elapsed_seconds": 0.0, "attribution_bytes": 0, "hashes_count": 0,
                "writers": {s: {"shards": 0, "tokens": 0} for s in self.writers},
                "languages": {lang: _new_lang_state() for lang in self.cfg.languages}}

    def _checkpoint(self) -> None:
        """Persist everything, state file last (see module docstring for crash safety)."""
        for split in self.writers:
            self._flush_split(split)
        ckpt = self.state["ckpt"] + 1
        for w in self.writers.values():
            w.checkpoint(ckpt)
        self.state["attribution_bytes"] = self.attrib.flush()
        self.state["hashes_count"] = self.hashes.flush()
        for split, w in self.writers.items():
            self.state["writers"][split] = {"shards": len(w.shards),
                                            "tokens": w.total_tokens}
        now = time.time()
        self.state["elapsed_seconds"] += now - self.session_t0
        self.session_t0 = now
        self.state["ckpt"] = ckpt
        tmp = self.cfg.out / "build-state.json.tmp"
        tmp.write_text(json.dumps(self.state, indent=2))
        os.replace(tmp, self.state_path)
        for w in self.writers.values():
            w.drop_pending_files(keep_ckpt=ckpt)

    # -- pipeline ---------------------------------------------------------

    def _flush_split(self, split: str) -> None:
        batch = self.batches[split]
        if not batch:
            return
        eot = np.array([self.tok.eot_token], dtype=np.uint16)
        parts: list[np.ndarray] = []
        ids_batch = self.tok.encode_batch([text for text, _ in batch])
        for (_, rec), ids in zip(batch, ids_batch, strict=True):
            parts.append(np.asarray(ids, dtype=np.uint16))
            parts.append(eot)
            rec["split"] = split
            rec["tokens"] = len(ids) + 1
            self.attrib.add((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
            self.state["languages"][rec["lang"]][f"{split}_tokens"] += len(ids) + 1
        self.writers[split].append(np.concatenate(parts))
        self.batches[split] = []

    def _is_val(self, digest: bytes) -> bool:
        if "val" not in self.writers:
            return False
        return int.from_bytes(digest[8:12], "big") / 2**32 < self.cfg.val_fraction

    def _process_row(self, row: SourceRow, lang: str) -> None:
        ls = self.state["languages"][lang]
        ls["rows"] += 1
        expected = STACK_LANG_NAMES.get(lang)
        if expected is not None and row.lang != expected:
            raise RuntimeError(f"language partition mismatch for {lang!r}: row.lang="
                               f"{row.lang!r} (file {row.file_idx} row {row.row_idx})")
        # Markdown is prose: the code rules flag unwrapped natural-text paragraphs as
        # minified blobs, so it takes the gate's PROSE path instead.
        text, reason = clean_one(row.content, code=lang not in PROSE_LANGS,
                                 min_chars=self.cfg.min_chars,
                                 max_chars=self.cfg.max_chars)
        if reason is not None:
            ls[f"dropped_{reason}"] += 1
            return
        if not license_ok(row.licenses):
            ls["dropped_license"] += 1
            return
        if not (row.hexsha and row.repo and row.path):
            ls["dropped_attribution"] += 1  # unattributable file: cannot ship it
            return
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        if not self.hashes.add(digest[:8]):
            ls["dropped_dup"] += 1
            return
        if near_dup_reason(text) is not None:
            ls["dropped_near_dup"] += 1
            return
        split = "val" if self._is_val(digest) else "train"
        rec = {"lang": lang, "repo": row.repo, "hexsha": row.hexsha, "path": row.path,
               "licenses": list(row.licenses)}
        ls["kept_docs"] += 1
        self.batches[split].append((text, rec))
        if len(self.batches[split]) >= self.cfg.batch_docs:
            self._flush_split(split)

    def _lang_tokens(self, lang: str) -> int:
        ls = self.state["languages"][lang]
        return ls["train_tokens"] + ls["val_tokens"]

    def _progress(self, lang: str, budget: int) -> None:
        ls = self.state["languages"][lang]
        total = sum(self._lang_tokens(la) for la in self.cfg.languages)
        elapsed = self.state["elapsed_seconds"]
        rate = total / elapsed * 3600 if elapsed > 0 else 0.0
        drops = " ".join(f"{k.removeprefix('dropped_')}={ls[k]}" for k in DROP_KEYS)
        print(f"[{lang}] rows={ls['rows']:,} kept={ls['kept_docs']:,} "
              f"tokens={self._lang_tokens(lang):,}/{budget:,} ({drops}) "
              f"| overall {total:,} tok @ {rate:,.0f} tok/h", flush=True)

    def run(self) -> dict:
        if self.state["completed"]:
            print(f"{self.state_path} says the build is complete; nothing to do")
            return self.state
        wsum = sum(self.cfg.weights)
        budgets = [int(self.cfg.target_tokens * w / wsum) for w in self.cfg.weights]
        download_dir = self.cfg.out / "_download"
        for lang, budget in zip(self.cfg.languages, budgets, strict=True):
            ls = self.state["languages"][lang]
            if ls["done"]:
                continue
            print(f"[{lang}] budget {budget:,} tokens; starting at file {ls['file_idx']} "
                  f"row {ls['rows_done']}", flush=True)
            rows = self.rows_factory(lang, download_dir=download_dir,
                                     start_file=ls["file_idx"], start_row=ls["rows_done"])
            since_ckpt = 0
            for row in rows:
                self._process_row(row, lang)
                ls["file_idx"], ls["rows_done"] = row.file_idx, row.row_idx + 1
                since_ckpt += 1
                if self._lang_tokens(lang) >= budget:
                    break
                if since_ckpt >= self.cfg.checkpoint_rows:
                    self._checkpoint()
                    self._progress(lang, budget)
                    since_ckpt = 0
            else:
                ls["exhausted"] = True  # source ran out before the budget was met
            rows.close()  # release the current download before the dir is cleaned up
            ls["done"] = True
            self._checkpoint()
            self._progress(lang, budget)
        for w in self.writers.values():
            w.finalize()
        self.state["completed"] = True
        self._checkpoint()  # records final shard counts; completed=True skips resume
        for w in self.writers.values():
            w.drop_pending_files(keep_ckpt=None)
        if download_dir.exists():
            shutil.rmtree(download_dir)
        self._summary(budgets)
        return self.state

    def _summary(self, budgets: list[int]) -> None:
        total = sum(self._lang_tokens(lang) for lang in self.cfg.languages)
        elapsed = self.state["elapsed_seconds"]
        print(f"\nbuild complete: {total:,} tokens in {elapsed/3600:.2f} h "
              f"({total/elapsed*3600:,.0f} tok/h)" if elapsed else "\nbuild complete")
        for split, w in self.writers.items():
            print(f"  {split}: {w.total_tokens:,} tokens in {len(w.shards)} shards "
                  f"({w.manifest_path})")
        print(f"  attribution manifest: {self.attrib.path} "
              f"({self.state['attribution_bytes']:,} bytes)")
        for lang, budget in zip(self.cfg.languages, budgets, strict=True):
            ls = self.state["languages"][lang]
            note = " [SOURCE EXHAUSTED before budget]" if ls["exhausted"] else ""
            print(f"  {lang}: {self._lang_tokens(lang):,}/{budget:,} tokens, "
                  f"{ls['kept_docs']:,} docs kept of {ls['rows']:,} rows{note}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="corpus dir (shards + manifests + "
                    "attribution.jsonl + build-state.json)")
    ap.add_argument("--tokenizer", required=True,
                    help="tokenizer json, e.g. data/tokenizers/code-49k.json")
    ap.add_argument("--languages", nargs="+", default=["python", "javascript", "typescript"],
                    choices=sorted(STACK_DIRS))
    ap.add_argument("--target-tokens", type=int, required=True,
                    help="total token budget across languages (train+val)")
    ap.add_argument("--val-fraction", type=float, default=0.001,
                    help="deterministic content-hash fraction routed to the val split")
    ap.add_argument("--lang-weights", nargs="+", type=float, default=None,
                    help="per-language budget weights (default: equal)")
    ap.add_argument("--shard-size", type=int, default=100_000_000)
    ap.add_argument("--batch-docs", type=int, default=512)
    ap.add_argument("--checkpoint-rows", type=int, default=50_000)
    ap.add_argument("--min-chars", type=int, default=64)
    ap.add_argument("--max-chars", type=int, default=1_000_000)
    ap.add_argument("--near-dedup", default=None,
                    help="near-dedup selector -- NOT IMPLEMENTED, raises "
                         "(see near_dup_reason for the hook)")
    args = ap.parse_args()

    if args.near_dedup is not None:
        raise NotImplementedError(
            "near-dedup is a documented hook only (near_dup_reason); exact sha256 "
            "dedup is what exists today")
    weights = args.lang_weights if args.lang_weights is not None else [1.0] * len(args.languages)
    if len(weights) != len(args.languages):
        raise ValueError(f"--lang-weights needs {len(args.languages)} values, "
                         f"got {len(weights)}")
    cfg = BuildConfig(out=Path(args.out), tokenizer=args.tokenizer,
                      languages=args.languages, target_tokens=args.target_tokens,
                      val_fraction=args.val_fraction, weights=weights,
                      shard_size=args.shard_size, batch_docs=args.batch_docs,
                      checkpoint_rows=args.checkpoint_rows, min_chars=args.min_chars,
                      max_chars=args.max_chars)
    CorpusBuilder(cfg).run()


if __name__ == "__main__":
    main()
