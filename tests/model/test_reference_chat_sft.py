"""Multi-turn chat SFT example building (model/reference/chat_sft.py): the conversation
template must stay byte-compatible with the single-turn recipe (format_chat + "\n### End")
so the serving stop-strings keep working, every assistant turn (not just the last) is
supervised, and truncation drops whole LEADING turns so the most recent context survives."""

from __future__ import annotations

import pytest

from microlab.model.reference.chat_sft import (
    END_SENTINEL,
    TurnTooLongError,
    build_chat_example,
    render_conversation,
)
from microlab.model.reference.sft import IGNORE_INDEX, build_sft_example, format_chat


class _ByteTok:
    """Byte-level tokenizer: encode is exact per char, so segment boundaries don't shift."""

    def encode(self, s):
        return list(s.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8")


def test_end_sentinel_is_the_recipe_sentinel():
    assert END_SENTINEL == "\n### End"


def test_render_two_turns_exact_template():
    # Interior turns end "### End\n" (the \n separates blocks and belongs to the NEXT,
    # masked, prompt segment); the final turn ends bare "### End" exactly like single-turn.
    segments = render_conversation([
        {"user": "U1", "assistant": "A1"},
        {"user": "U2", "assistant": "A2"},
    ])
    text = "".join(p + r for p, r in segments)
    assert text == (
        "### Instruction:\nU1\n\n### Response:\nA1\n### End\n"
        "### Instruction:\nU2\n\n### Response:\nA2\n### End"
    )
    # Serving stops on "### End" and "\n### Instruction:"; both boundaries must appear
    # verbatim at the turn seam so a served model's stop strings still fire.
    assert "### End\n### Instruction:" in text


def test_render_single_turn_matches_format_chat():
    ((prompt, response),) = render_conversation([{"user": "Say hi", "assistant": "hello"}])
    expect_prompt, _ = format_chat("Say hi", "")
    assert prompt == expect_prompt
    assert response == "hello" + END_SENTINEL


def test_render_context_uses_input_block():
    # Optional per-turn "context" renders through format_chat's "### Input:" block, so
    # sft_mix rows with context keep their exact single-turn layout inside a conversation.
    ((prompt, _),) = render_conversation(
        [{"user": "summarize", "context": "the ctx", "assistant": "ok"}])
    assert prompt == format_chat("summarize", "the ctx")[0]


def test_single_turn_example_byte_identical_to_build_sft_example():
    tok = _ByteTok()
    for turn in ({"user": "Say hi", "assistant": "hello"},
                 {"user": "summarize", "context": "the ctx", "assistant": "ok"}):
        ids, labels, dropped = build_chat_example(tok, [turn], block_size=4096)
        prompt, _ = format_chat(turn["user"], turn.get("context", ""))
        want_ids, want_labels = build_sft_example(tok, prompt, turn["assistant"] + END_SENTINEL)
        assert ids == want_ids
        assert labels == want_labels
        assert dropped == 0


def test_all_assistant_turns_supervised_prompts_masked():
    tok = _ByteTok()
    turns = [{"user": "U1", "assistant": "A1"}, {"user": "U2", "assistant": "A2"}]
    ids, labels, dropped = build_chat_example(tok, turns, block_size=4096)
    assert dropped == 0
    # Rebuild the expected label layout segment by segment: prompts (incl. the "\n" turn
    # separator) masked, every response + sentinel supervised.
    expect_ids: list[int] = []
    expect_labels: list[int] = []
    for prompt, response in render_conversation(turns):
        p_ids, r_ids = tok.encode(prompt), tok.encode(response)
        expect_ids += p_ids + r_ids
        expect_labels += [IGNORE_INDEX] * len(p_ids) + r_ids
    assert ids == expect_ids
    assert labels == expect_labels
    # Both turns' responses appear in the supervised positions (not just the last one).
    supervised = tok.decode([t for t in labels if t != IGNORE_INDEX])
    assert "A1" in supervised and "A2" in supervised


def test_truncation_drops_whole_leading_turns():
    tok = _ByteTok()
    turns = [{"user": "U1" * 30, "assistant": "A1" * 30},
             {"user": "U2", "assistant": "A2"},
             {"user": "U3", "assistant": "A3"}]
    full_len = len(build_chat_example(tok, turns, block_size=10_000)[0])
    # A budget below the full conversation but able to hold the last two turns.
    small = len(build_chat_example(tok, turns[1:], block_size=10_000)[0])
    assert small < full_len
    ids, labels, dropped = build_chat_example(tok, turns, block_size=small)
    want_ids, want_labels, _ = build_chat_example(tok, turns[1:], block_size=small)
    assert dropped == 1
    assert ids == want_ids and labels == want_labels
    # After dropping, the window is re-rendered from scratch: no leading "\n" separator.
    assert tok.decode(ids).startswith("### Instruction:\nU2")


def test_raises_when_even_last_turn_does_not_fit():
    tok = _ByteTok()
    turns = [{"user": "U", "assistant": "A" * 200}]
    with pytest.raises(TurnTooLongError):
        build_chat_example(tok, turns, block_size=64)


def test_invalid_conversations_raise():
    tok = _ByteTok()
    with pytest.raises(ValueError):
        build_chat_example(tok, [], block_size=64)
    with pytest.raises(ValueError):
        build_chat_example(tok, [{"user": "  ", "assistant": "a"}], block_size=64)
    with pytest.raises(ValueError):
        build_chat_example(tok, [{"user": "u", "assistant": ""}], block_size=64)
