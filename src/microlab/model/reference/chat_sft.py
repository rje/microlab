"""Multi-turn chat SFT example building on top of the single-turn recipe (sft.py).

A conversation [{"user", "assistant", optional "context"}, ...] renders as repeated
single-turn blocks:

    ### Instruction:\n{user}\n\n### Response:\n{assistant}\n### End\n

with the final turn ending in a bare "\n### End" (no trailing newline) so a one-turn
conversation is BYTE-IDENTICAL to the existing single-turn recipe (format_chat +
END_SENTINEL) and the serving stop-strings ("### End", "\n### Instruction:") keep firing
at every turn seam. The interior "\n" after "### End" is the turn separator; it belongs
to the NEXT prompt segment and is therefore masked.

Masking: labels are IGNORE_INDEX everywhere EXCEPT every assistant response + its
"\n### End" sentinel. ALL turns are supervised, not just the last: each conversation then
contributes several response targets instead of one, so a multi-turn example carries far
more learning signal per sequence, and every supervised position still only conditions on
its left context, exactly like serving.

Truncation drops whole LEADING turns (the most recent turns are what a chat model must
condition on); a conversation whose final turn alone cannot fit raises TurnTooLongError
so callers can skip-and-count rather than silently train on a clipped response.
"""

from __future__ import annotations

from microlab.model.reference.sft import IGNORE_INDEX, format_chat

# Trained onto the end of every response so the model learns to emit it; serving stops here.
END_SENTINEL = "\n### End"

Turn = dict[str, str]


class TurnTooLongError(ValueError):
    """Even the final turn of the conversation cannot fit in the block size."""


def _validate(turns: list[Turn]) -> None:
    if not turns:
        raise ValueError("conversation has no turns")
    for i, turn in enumerate(turns):
        if not (turn.get("user") or "").strip():
            raise ValueError(f"turn {i}: empty user message")
        if not (turn.get("assistant") or "").strip():
            raise ValueError(f"turn {i}: empty assistant message")


def render_conversation(turns: list[Turn]) -> list[tuple[str, str]]:
    """Render a conversation as [(prompt_text, response_text), ...] per turn. Prompt
    segments reuse format_chat verbatim (single source of truth for the template); every
    prompt after the first is prefixed with the "\\n" separator that follows the previous
    turn's "### End". Response segments carry the END_SENTINEL."""
    _validate(turns)
    segments: list[tuple[str, str]] = []
    for i, turn in enumerate(turns):
        prompt, _ = format_chat(turn["user"], turn.get("context", ""))
        if i > 0:
            prompt = "\n" + prompt
        segments.append((prompt, turn["assistant"] + END_SENTINEL))
    return segments


def build_chat_example(
    tok, turns: list[Turn], block_size: int
) -> tuple[list[int], list[int], int]:
    """Tokenize a conversation into (input_ids, labels, dropped_turns) with every prompt
    segment masked to IGNORE_INDEX and every response (+ sentinel) supervised.

    If the full conversation exceeds block_size, whole leading turns are dropped until the
    remainder fits (the window is re-rendered so the new first turn has no separator).
    Raises TurnTooLongError when even the final turn alone does not fit."""
    _validate(turns)
    for start in range(len(turns)):
        input_ids: list[int] = []
        labels: list[int] = []
        for prompt, response in render_conversation(turns[start:]):
            p_ids = tok.encode(prompt)
            r_ids = tok.encode(response)
            input_ids += p_ids + r_ids
            labels += [IGNORE_INDEX] * len(p_ids) + list(r_ids)
        if len(input_ids) <= block_size:
            return input_ids, labels, start
    raise TurnTooLongError(
        f"final turn alone exceeds block_size={block_size} "
        f"({len(input_ids)} tokens)"
    )
