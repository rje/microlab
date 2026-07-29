"""Tool-call eval machinery: the shipped registry/items validate, prompt rendering is
deterministic and budget-shaped, reply parsing is robust, and the scoring math
(argument F1, split accounting) is exactly right."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from microlab.evals.code.toolcall import (
    ALWAYS_SHOWN,
    aggregate,
    argument_f1,
    build_prompt,
    load_items,
    load_registry,
    parse_call,
    render_tool,
    score_item,
    shown_tools,
)

REPO = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO / "evals" / "toolcall" / "registry.json"
ITEMS_PATH = REPO / "evals" / "toolcall" / "items.jsonl"


@pytest.fixture(scope="module")
def registry():
    return load_registry(REGISTRY_PATH)


@pytest.fixture(scope="module")
def items(registry):
    return load_items(ITEMS_PATH, registry)


# ---------------------------------------------------------------- shipped data validates

def test_shipped_registry_composition(registry):
    assert len(registry) >= 25
    for marker in ("clarify", "none"):
        assert marker in registry
        assert registry[marker]["params"] == []
    assert "get_time" in registry  # the few-shot demo tool


def test_shipped_items_counts_and_splits(items):
    assert len(items) == 120
    splits = [i["split"] for i in items]
    assert splits.count("pattern") == 60
    assert splits.count("compositional") == 60
    kinds = {i["kind"] for i in items}
    assert {"direct", "inference", "distractor", "clarify", "none"} <= kinds


def test_shipped_items_marker_cases_live_in_compositional(items):
    for it in items:
        if it["expected"]["tool"] in ("clarify", "none"):
            assert it["split"] == "compositional", it["id"]
            assert it["expected"]["arguments"] == {}, it["id"]


# ------------------------------------------------------------------- validation catches

def test_load_registry_rejects_missing_marker(tmp_path):
    bad = {"tools": [{"name": "get_time", "description": "d",
                      "params": [{"name": "timezone", "type": "str", "required": True}]},
                     {"name": "none", "description": "d", "params": []}]}
    p = tmp_path / "reg.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="clarify"):
        load_registry(p)


def test_load_items_rejects_unknown_expected_tool(tmp_path, registry):
    item = {"id": "x-1", "split": "pattern", "request": "r", "tools": ["get_weather"],
            "expected": {"tool": "launch_rocket", "arguments": {}}}
    p = tmp_path / "items.jsonl"
    p.write_text(json.dumps(item) + "\n")
    with pytest.raises(ValueError, match="not in registry"):
        load_items(p, registry)


def test_load_items_rejects_missing_required_arg(tmp_path, registry):
    item = {"id": "x-1", "split": "pattern", "request": "r", "tools": ["get_forecast"],
            "expected": {"tool": "get_forecast", "arguments": {"city": "Oslo"}}}  # no days
    p = tmp_path / "items.jsonl"
    p.write_text(json.dumps(item) + "\n")
    with pytest.raises(ValueError, match="required param 'days'"):
        load_items(p, registry)


def test_load_items_rejects_duplicate_ids(tmp_path, registry):
    item = {"id": "x-1", "split": "pattern", "request": "r", "tools": [],
            "expected": {"tool": "none", "arguments": {}}}
    p = tmp_path / "items.jsonl"
    p.write_text(json.dumps(item) + "\n" + json.dumps(item) + "\n")
    with pytest.raises(ValueError, match="duplicate item id"):
        load_items(p, registry)


# --------------------------------------------------------------------------- rendering

def test_render_tool_compact_line(registry):
    line = render_tool(registry["get_forecast"])
    assert line == ("- get_forecast(city: str, days: int) — "
                    "Weather forecast for a city over the coming days")
    assert "units?" in render_tool(registry["get_weather"])  # optional marked with ?


def test_shown_tools_always_include_markers_in_registry_order(items, registry):
    tools = shown_tools(items[0], registry)
    names = [t["name"] for t in tools]
    for name in ALWAYS_SHOWN:
        assert name in names
    order = list(registry)
    assert names == sorted(names, key=order.index)


def test_build_prompt_deterministic_and_shaped(items, registry):
    a = build_prompt(items[0], registry)
    assert a == build_prompt(items[0], registry)
    assert a.count("### Instruction:") == 3  # two few-shot turns + the item
    assert a.count("### End") == 2
    assert a.endswith("### Response:\n")
    assert items[0]["request"] in a


# ----------------------------------------------------------------------------- parsing

def test_parse_call_plain_and_with_prose():
    assert parse_call('{"tool": "get_weather", "arguments": {"city": "Paris"}}') == \
        ("get_weather", {"city": "Paris"})
    assert parse_call('Sure: {"tool": "none", "arguments": {}} done') == ("none", {})


def test_parse_call_defaults_missing_arguments():
    assert parse_call('{"tool": "get_news"}') == ("get_news", {})


def test_parse_call_braces_inside_string_values():
    reply = '{"tool": "send_sms", "arguments": {"to": "1", "message": "use {x} ok"}}'
    assert parse_call(reply) == ("send_sms", {"to": "1", "message": "use {x} ok"})


def test_parse_call_failures():
    assert parse_call("I would use the weather tool.") is None
    assert parse_call('{"tool": 3, "arguments": {}}') is None
    assert parse_call('{"tool": "x", "arguments": []}') is None
    assert parse_call('{"tool": "x", "arguments"') is None


# ----------------------------------------------------------------------------- scoring

def test_argument_f1_exact_and_empty():
    assert argument_f1({"a": 1}, {"a": 1}) == 1.0
    assert argument_f1({}, {}) == 1.0
    assert argument_f1({"a": 1}, {}) == 0.0
    assert argument_f1({}, {"a": 1}) == 0.0


def test_argument_f1_partial_and_extra():
    # one of two expected present: P=1, R=0.5 -> F1 2/3
    assert argument_f1({"a": 1, "b": 2}, {"a": 1}) == pytest.approx(2 / 3)
    # extra key: P=0.5, R=1 -> F1 2/3
    assert argument_f1({"a": 1}, {"a": 1, "b": 9}) == pytest.approx(2 / 3)
    assert argument_f1({"a": 1}, {"a": 2}) == 0.0


def test_argument_f1_normalization():
    assert argument_f1({"city": "Paris"}, {"city": "  paris "}) == 1.0
    assert argument_f1({"days": 5}, {"days": 5.0}) == 1.0
    assert argument_f1({"on": True}, {"on": True}) == 1.0
    assert argument_f1({"xs": ["A", "b"]}, {"xs": ["a", "B"]}) == 1.0
    # bool is not int-1 under normalization
    assert argument_f1({"on": True}, {"on": 1}) == 0.0


def _item(tool, args, split="pattern"):
    return {"id": "t-1", "split": split, "request": "r", "tools": [],
            "expected": {"tool": tool, "arguments": args}}


def test_score_item_strict_and_wrong_tool():
    it = _item("get_weather", {"city": "Paris"})
    good = score_item(it, '{"tool": "get_weather", "arguments": {"city": "Paris"}}')
    assert good["strict"] and good["tool_correct"] and good["arg_f1"] == 1.0
    wrong = score_item(it, '{"tool": "get_forecast", "arguments": {"city": "Paris"}}')
    assert not wrong["tool_correct"] and wrong["arg_f1"] == 0.0 and not wrong["strict"]


def test_score_item_unparseable():
    r = score_item(_item("none", {}), "the weather one, probably")
    assert not r["parsed"] and not r["strict"] and r["arg_f1"] == 0.0


def test_score_item_clarify_marker():
    r = score_item(_item("clarify", {}), '{"tool": "clarify", "arguments": {}}')
    assert r["strict"] and r["arg_f1"] == 1.0


def test_aggregate_split_accounting():
    results = [
        score_item(_item("get_news", {}, "pattern"),
                   '{"tool": "get_news", "arguments": {}}'),
        score_item(_item("get_news", {"topic": "a"}, "pattern"),
                   '{"tool": "get_news", "arguments": {}}'),  # tool right, args wrong
        score_item(_item("none", {}, "compositional"),
                   '{"tool": "get_news", "arguments": {}}'),  # wrong tool
        score_item(_item("none", {}, "compositional"), "unparseable"),
    ]
    agg = aggregate(results)
    assert agg["overall"]["n"] == 4
    assert agg["overall"]["tool_acc"] == 0.5
    assert agg["overall"]["strict_acc"] == 0.25
    assert agg["overall"]["parse_failures"] == 1
    assert agg["splits"]["pattern"]["n"] == 2
    assert agg["splits"]["pattern"]["tool_acc"] == 1.0
    assert agg["splits"]["pattern"]["strict_acc"] == 0.5
    assert agg["splits"]["pattern"]["arg_f1_when_tool_correct"] == 0.5
    assert agg["splits"]["compositional"]["tool_acc"] == 0.0
    assert agg["splits"]["compositional"]["arg_f1_when_tool_correct"] == 0.0


def test_aggregate_empty_raises():
    with pytest.raises(ValueError, match="no results"):
        aggregate([])
