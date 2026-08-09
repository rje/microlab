"""Pure, network-free helpers for building the coder-1b code-instruction SFT mixes.

Every function here is deterministic and importable without torch or a GPU so the builders
in scripts/ stay thin and the logic is unit-tested off-network — the same split
build_sft_mix.py uses. Row is the single-turn schema scripts/sft.py consumes.
"""
from __future__ import annotations

import importlib.util as _ilu
import re as _re
from pathlib import Path as _Path

Row = dict[str, str]

# Reuse the OASST tree-walker from the chat-mix builder (single source of truth for the
# rank-0-child linearization); scripts/ isn't a package so load it by path.
_bcm_spec = _ilu.spec_from_file_location(
    "build_chat_mix", _Path(__file__).resolve().parents[3] / "scripts" / "build_chat_mix.py")
_bcm = _ilu.module_from_spec(_bcm_spec)
_bcm_spec.loader.exec_module(_bcm)

_CODE_FENCE = _re.compile(r"```")


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
    # Responses are stripped to match the other normalizers (normalize_alpaca /
    # normalize_no_robots) and because END_SENTINEL ("\n### End") supplies the trailing boundary.
    response = (row.get("new_contents") or "").strip()
    if not instruction or not response:
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


def is_code_conv(conv: dict) -> bool:
    """True if any assistant turn contains a fenced code block (```). The cheap, precise
    signal that a thread is about code without language-classifying every message."""
    return any(_CODE_FENCE.search(t.get("assistant", "")) for t in conv.get("turns", []))


def oasst_code_convs(messages: list[dict], max_turns: int = 6) -> list[dict]:
    """Linearize OASST trees (best-ranked child) and keep only code-bearing conversations."""
    convs = _bcm.extract_oasst_conversations(messages, max_turns=max_turns)
    return [c for c in convs if is_code_conv(c)]
