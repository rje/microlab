"""Source-agnostic corpus pipeline: clean -> exact-dedup -> contamination-filter ->
train/val/test split. Deterministic (seeded) so tests and runs reproduce."""

from __future__ import annotations

import hashlib
import random
import re
import unicodedata


def clean_text(text: str) -> str:
    """Normalize unicode (NFC), strip control chars except \\n and \\t, collapse
    runs of spaces, and trim trailing whitespace per line."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(
        c for c in text if c == "\n" or c == "\t" or not unicodedata.category(c).startswith("C")
    )
    text = re.sub(r" +", " ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text


def dedup_docs(docs: list[str]) -> list[str]:
    """Exact dedup by content hash, preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for d in docs:
        h = hashlib.sha256(d.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(d)
    return out


def filter_contamination(docs: list[str], eval_strings: list[str]) -> list[str]:
    """Drop any doc that contains any eval prompt verbatim (the contamination check)."""
    bad = [e for e in eval_strings if e]
    return [d for d in docs if not any(e in d for e in bad)]


def split_docs(
    docs: list[str], ratios: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 0
) -> dict[str, list[str]]:
    """Deterministic shuffle + split into train/val/test. Splits are disjoint."""
    assert abs(sum(ratios) - 1.0) < 1e-9, "ratios must sum to 1"
    shuffled = list(docs)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }
