"""scripts/build_chat_mix.py: pure normalizers + the OASST tree walk + composition math.
No network in CI — the tree-walking logic (ranking selection, alternation enforcement,
deleted/empty/non-English dropping, prompter branching, depth cap) is what can silently
go wrong. Loaded via importlib since scripts/ isn't a package."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "build_chat_mix", Path(__file__).resolve().parents[2] / "scripts" / "build_chat_mix.py")
bcm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bcm)


def _msg(mid, parent, role, text, lang="en", rank=None, deleted=False, review=True):
    return {"message_id": mid, "parent_id": parent, "role": role, "text": text,
            "lang": lang, "rank": rank, "deleted": deleted, "review_result": review}


# ---------------------------------------------------------------- no_robots multi-turn


def test_no_robots_keeps_only_multi_turn():
    msgs = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]
    got = bcm.normalize_no_robots_chat({"messages": msgs})
    assert got == {"turns": [{"user": "q1", "assistant": "a1"},
                             {"user": "q2", "assistant": "a2"}]}
    # single-turn rows are already in sft_mix (build_sft_mix keeps them) -> dropped here
    assert bcm.normalize_no_robots_chat({"messages": msgs[:2]}) is None


def test_no_robots_folds_leading_system_into_first_turn_context():
    # EVERY multi-turn no_robots row opens with a system persona message; it conditions
    # the whole conversation, so it's kept as the first turn's "context" (rendered via the
    # template's existing "### Input:" block), not silently thrown away.
    msgs = [{"role": "system", "content": "You are a pirate chatbot."},
            {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]
    got = bcm.normalize_no_robots_chat({"messages": msgs})
    assert got == {"turns": [
        {"user": "q1", "context": "You are a pirate chatbot.", "assistant": "a1"},
        {"user": "q2", "assistant": "a2"}]}
    # an EMPTY system message is tolerated but adds no context key
    msgs[0]["content"] = " "
    assert bcm.normalize_no_robots_chat({"messages": msgs}) == \
        {"turns": [{"user": "q1", "assistant": "a1"}, {"user": "q2", "assistant": "a2"}]}


def test_no_robots_rejects_non_alternating():
    ok = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    # a system message anywhere but the front, or any broken alternation -> dropped
    assert bcm.normalize_no_robots_chat(
        {"messages": ok + [{"role": "system", "content": "s"}] + ok}) is None
    assert bcm.normalize_no_robots_chat(
        {"messages": ok + [{"role": "assistant", "content": "again"}, ok[0], ok[1]]}) is None
    # trailing user message with no reply -> dropped (must end on an assistant turn)
    assert bcm.normalize_no_robots_chat(
        {"messages": ok * 2 + [{"role": "user", "content": "dangling"}]}) is None
    # empty content anywhere -> dropped
    assert bcm.normalize_no_robots_chat(
        {"messages": ok + [{"role": "user", "content": " "},
                           {"role": "assistant", "content": "a2"}]}) is None


def test_no_robots_caps_depth():
    msgs = []
    for i in range(8):
        msgs += [{"role": "user", "content": f"q{i}"},
                 {"role": "assistant", "content": f"a{i}"}]
    got = bcm.normalize_no_robots_chat({"messages": msgs})
    assert len(got["turns"]) == bcm.MAX_TURNS == 6
    assert got["turns"][-1] == {"user": "q5", "assistant": "a5"}


# ---------------------------------------------------------------- sft_mix single-turn


def test_sft_row_to_conv_preserves_context():
    assert bcm.sft_row_to_conv({"instruction": "i", "context": "", "response": "r"}) == \
        {"turns": [{"user": "i", "assistant": "r"}]}
    assert bcm.sft_row_to_conv({"instruction": "i", "context": "c", "response": "r"}) == \
        {"turns": [{"user": "i", "context": "c", "assistant": "r"}]}
    assert bcm.sft_row_to_conv({"instruction": "i", "context": "", "response": " "}) is None
    assert bcm.sft_row_to_conv({"instruction": "", "context": "", "response": "r"}) is None


# ---------------------------------------------------------------- OASST tree walk


def test_oasst_linear_tree_extracts_one_conversation():
    msgs = [_msg("m1", None, "prompter", "q1"),
            _msg("m2", "m1", "assistant", "a1"),
            _msg("m3", "m2", "prompter", "q2"),
            _msg("m4", "m3", "assistant", "a2")]
    convs = bcm.extract_oasst_conversations(msgs)
    assert convs == [{"turns": [{"user": "q1", "assistant": "a1"},
                                {"user": "q2", "assistant": "a2"}]}]


def test_oasst_picks_highest_ranked_assistant_child():
    # rank 0 is best; the walk must take it even when listed later / with higher ids
    msgs = [_msg("m1", None, "prompter", "q"),
            _msg("m2", "m1", "assistant", "worse", rank=1.0),
            _msg("m3", "m1", "assistant", "best", rank=0.0)]
    assert bcm.extract_oasst_conversations(msgs) == \
        [{"turns": [{"user": "q", "assistant": "best"}]}]
    # rank None ranks WORSE than any ranked sibling
    msgs = [_msg("m1", None, "prompter", "q"),
            _msg("m2", "m1", "assistant", "unranked", rank=None),
            _msg("m3", "m1", "assistant", "ranked", rank=2.0)]
    assert bcm.extract_oasst_conversations(msgs) == \
        [{"turns": [{"user": "q", "assistant": "ranked"}]}]
    # all None -> deterministic pick by message_id
    msgs = [_msg("m1", None, "prompter", "q"),
            _msg("m9", "m1", "assistant", "later-id", rank=None),
            _msg("m2", "m1", "assistant", "earlier-id", rank=None)]
    assert bcm.extract_oasst_conversations(msgs) == \
        [{"turns": [{"user": "q", "assistant": "earlier-id"}]}]


def test_oasst_drops_deleted_empty_and_non_english():
    msgs = [_msg("m1", None, "prompter", "q1"),
            _msg("m2", "m1", "assistant", "a1", rank=0.0, deleted=True),   # deleted best
            _msg("m3", "m1", "assistant", "a1-alt", rank=1.0),
            _msg("m4", "m3", "prompter", "  "),                            # empty -> path ends
            _msg("m5", "m4", "assistant", "unreachable")]
    assert bcm.extract_oasst_conversations(msgs) == \
        [{"turns": [{"user": "q1", "assistant": "a1-alt"}]}]
    # a non-English root yields nothing; failed review is dropped like deleted
    assert bcm.extract_oasst_conversations([_msg("m1", None, "prompter", "hola", lang="es")]) == []
    msgs = [_msg("m1", None, "prompter", "q"),
            _msg("m2", "m1", "assistant", "bad", rank=0.0, review=False),
            _msg("m3", "m1", "assistant", "good", rank=1.0)]
    assert bcm.extract_oasst_conversations(msgs) == \
        [{"turns": [{"user": "q", "assistant": "good"}]}]


def test_oasst_enforces_alternation():
    # a same-role child (prompter under prompter) is never followed; an assistant root or a
    # root with no assistant reply yields no conversation (nothing supervisable)
    msgs = [_msg("m1", None, "prompter", "q"),
            _msg("m2", "m1", "prompter", "not-a-reply")]
    assert bcm.extract_oasst_conversations(msgs) == []
    assert bcm.extract_oasst_conversations([_msg("m1", None, "assistant", "hi")]) == []


def test_oasst_prompter_branching_yields_multiple_conversations():
    # two different follow-up questions under one reply are genuinely different
    # conversations (unlike ranked assistant variants) -> both paths are extracted
    msgs = [_msg("m1", None, "prompter", "q1"),
            _msg("m2", "m1", "assistant", "a1", rank=0.0),
            _msg("m3", "m2", "prompter", "follow-A"),
            _msg("m4", "m2", "prompter", "follow-B"),
            _msg("m5", "m3", "assistant", "reply-A", rank=0.0),
            _msg("m6", "m4", "assistant", "reply-B", rank=0.0)]
    convs = bcm.extract_oasst_conversations(msgs)
    assert len(convs) == 2
    seconds = sorted(c["turns"][1]["user"] for c in convs)
    assert seconds == ["follow-A", "follow-B"]
    assert all(c["turns"][0] == {"user": "q1", "assistant": "a1"} for c in convs)


def test_oasst_caps_depth_at_max_turns():
    msgs = []
    parent = None
    for i in range(9):  # 9 user/assistant pairs in a linear chain
        msgs.append(_msg(f"p{i}", parent, "prompter", f"q{i}"))
        msgs.append(_msg(f"a{i}", f"p{i}", "assistant", f"a{i}", rank=0.0))
        parent = f"a{i}"
    (conv,) = bcm.extract_oasst_conversations(msgs)
    assert len(conv["turns"]) == bcm.MAX_TURNS == 6
    assert conv["turns"][-1] == {"user": "q5", "assistant": "a5"}


def test_oasst_dedupes_identical_conversations():
    # same text under different ids (dataset dupes) must not produce duplicate examples
    msgs = [_msg("m1", None, "prompter", "q"), _msg("m2", "m1", "assistant", "a", rank=0.0),
            _msg("n1", None, "prompter", "q"), _msg("n2", "n1", "assistant", "a", rank=0.0)]
    assert bcm.extract_oasst_conversations(msgs) == \
        [{"turns": [{"user": "q", "assistant": "a"}]}]


# ---------------------------------------------------------------- composition


def _convs(n, tag, turns=2):
    return [{"turns": [{"user": f"{tag}{i}u{t}", "assistant": f"{tag}{i}a{t}"}
                       for t in range(turns)]} for i in range(n)]


def test_compose_mix_hits_target_fraction():
    multi = _convs(4, "m")
    filler = _convs(100, "s", turns=1)
    mix, counts = bcm.compose_mix(multi, [], filler, multi_frac=0.4, seed=0)
    assert counts["multi"] == 4 and counts["single"] == 6 and counts["total"] == 10
    got_multi = [c for c in mix if len(c["turns"]) > 1]
    assert len(got_multi) == 4
    # deterministic under the same seed
    mix2, _ = bcm.compose_mix(multi, [], filler, multi_frac=0.4, seed=0)
    assert mix == mix2


def test_compose_mix_keeps_priority_singles_and_fills_from_pool():
    # 1-turn conversations extracted from OASST are scarce human data: ALL kept, and only
    # the remainder of the single budget is sampled from the (huge) sft_mix filler pool.
    multi = _convs(4, "m")                      # single budget at 0.4 -> 6
    keep = _convs(2, "k", turns=1)
    filler = _convs(100, "s", turns=1)
    mix, counts = bcm.compose_mix(multi, keep, filler, multi_frac=0.4, seed=0)
    assert counts == {"multi": 4, "single": 6, "single_kept": 2, "single_filled": 4,
                      "total": 10}
    users = {c["turns"][0]["user"] for c in mix}
    assert {"k0u0", "k1u0"} <= users


def test_compose_mix_uses_all_filler_when_short_but_raises_out_of_band():
    multi = _convs(10, "m")
    # 12 filler singles available: all used -> frac 10/22 = 0.45, inside [0.3, 0.5]
    mix, counts = bcm.compose_mix(multi, [], _convs(12, "s", turns=1), multi_frac=0.4, seed=0)
    assert counts["single"] == 12
    # 5 singles: frac 10/15 = 0.67 > 0.5 -> raise, don't quietly ship a skewed mix
    with pytest.raises(ValueError):
        bcm.compose_mix(multi, [], _convs(5, "s", turns=1), multi_frac=0.4, seed=0)
    # more kept singles than the budget allows -> frac below 0.3 -> raise as well
    with pytest.raises(ValueError):
        bcm.compose_mix(_convs(2, "m"), _convs(50, "k", turns=1), [], multi_frac=0.4, seed=0)


def test_turns_histogram():
    mix = _convs(2, "a", turns=1) + _convs(3, "b", turns=2) + _convs(1, "c", turns=6)
    assert bcm.turns_histogram(mix) == {1: 2, 2: 3, 6: 1}


def test_write_jsonl_roundtrips(tmp_path):
    rows = _convs(3, "x")
    out = tmp_path / "sub" / "chat_mix.jsonl"
    assert bcm.write_jsonl(rows, out) == 3
    assert [json.loads(line) for line in out.read_text().splitlines()] == rows
