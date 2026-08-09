"""Pure, network-free helpers for building the coder-1b code-instruction SFT mixes.

Every function here is deterministic and importable without torch or a GPU so the builders
in scripts/ stay thin and the logic is unit-tested off-network — the same split
build_sft_mix.py uses. Row is the single-turn schema scripts/sft.py consumes.
"""
from __future__ import annotations

Row = dict[str, str]


def normalize_commitpack(row: dict, lang_allow: set[str] | None) -> Row | None:
    """CommitPackFT row -> {instruction=commit message, response=new file contents}.

    The commit message is the instruction; the post-commit file is the target. `lang_allow`
    (lowercased language names) gates languages — Python-first for this run. Returns None for
    a disallowed language or an empty message/body.
    """
    lang = (row.get("lang") or "").strip().lower()
    if lang_allow is not None and lang not in lang_allow:
        return None
    # CommitPackFT uses `message`; `subject` is the first line. Prefer the subject as the
    # instruction (concise), fall back to the full message.
    instruction = (row.get("subject") or row.get("message") or "").strip()
    response = row.get("new_contents") or ""
    if not instruction or not response.strip():
        return None
    return {"instruction": instruction, "context": "", "response": response}


def normalize_mbpp_train(row: dict) -> Row | None:
    """MBPP (sanitized) train/validation/prompt row -> {instruction=text, response=code}.

    NOT the test split (that is the eval set). Returns None if text or code is empty.
    """
    instruction = (row.get("text") or row.get("prompt") or "").strip()
    response = (row.get("code") or "").strip()
    if not instruction or not response:
        return None
    return {"instruction": instruction, "context": "", "response": response}
