"""Corpus source loaders. `load_sample()` reads the bundled public-domain text (used
by tests, offline). The others are recipes for real corpora pulled when training.

Sourcing ladder (license-clean):
  1. TinyShakespeare (~1 MB, public domain) — bring-up / first run.
  2. Project Gutenberg subset (public domain) or WikiText-103 (CC-BY-SA) — real corpus.
  3. TinyStories (permissive, HF) — pretraining where a 10-30M model is actually fluent.
"""

from __future__ import annotations

from pathlib import Path

_SAMPLE = Path(__file__).with_name("sample.txt")


def load_sample() -> str:
    """The bundled public-domain sample corpus (offline; used by tests)."""
    return _SAMPLE.read_text(encoding="utf-8")


def load_tinyshakespeare(path: str) -> str:
    """TinyShakespeare is a single ~1 MB public-domain text file; pass its local path.

    Get it once with, e.g.:
        curl -L -o tinyshakespeare.txt \
          https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    """
    return Path(path).read_text(encoding="utf-8")


def load_hf_text(
    dataset: str, split: str = "train", text_field: str = "text", **kwargs
) -> list[str]:
    """Recipe for HF corpora. Requires the optional `datasets` dep; not run in CI.

    Examples:
        load_hf_text("roneneldan/TinyStories")
        load_hf_text("wikitext", name="wikitext-103-raw-v1")
    """
    from datasets import load_dataset  # local import: optional/heavy dep

    ds = load_dataset(dataset, split=split, **kwargs)
    return [row[text_field] for row in ds if row[text_field].strip()]


def load_text_file(path: str) -> str:
    """Read any local UTF-8 text corpus (e.g. a domain corpus for continued pretraining)."""
    from pathlib import Path

    return Path(path).read_text(encoding="utf-8")


def load_dolly(path: str, limit: int | None = None) -> list[dict[str, str]]:
    """Load Databricks Dolly-15k JSONL (CC-BY-SA) as {instruction, context, response} dicts."""
    import json
    from pathlib import Path

    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append(
            {"instruction": r.get("instruction", ""), "context": r.get("context", ""),
             "response": r.get("response", "")}
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows
