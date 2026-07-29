"""Tool-call eval: registry loading/validation, compact prompt rendering, response
parsing, and scoring (exact tool match + argument F1).

Prompt design (token budget). The 350M chat model has block_size 1024, so the whole
prompt must fit 1024 - max_new tokens. Rendering all ~25 registry tools would not fit,
so each eval item names the SUBSET of tools shown in its prompt (expected tool +
distractors — realistic anyway: production routers show a retrieval-filtered tool list).
The rendering is one line per tool, `name(param: type, optional?: type) — description`,
roughly 20-30 BPE tokens each. With <= 8 item tools + the always-present demo/marker
tools, directions (~90 tokens), two few-shot turns (~130 tokens) and the request, the
prompt lands around 500-700 tokens; scripts/eval_toolcall.py measures the real length
per item with the run's tokenizer and RAISES if prompt + max_new exceeds the block.

Few-shot: two fixed demonstration turns rendered with the trained multi-turn chat
template (repeated single-turn blocks with "### End" seams). Turn one calls `get_time`,
turn two answers `clarify` — so `get_time`, `clarify` and `none` are appended to every
shown tool list. Fixed examples + greedy decoding keep the whole eval deterministic.

Markers: `clarify` (a required argument is missing and must be asked for) and `none`
(no listed tool can satisfy the request) are pseudo-tools in the registry; the expected
"call" is exactly {"tool": "clarify"/"none", "arguments": {}}.

Scoring per item:
- tool_correct: predicted tool name == expected (unparseable replies are wrong);
- argument F1 over key/value pairs, a pair matching when the key exists and the
  normalized values are equal (strings casefolded/stripped, int-valued floats unified,
  lists/dicts recursively normalized); both-empty scores 1.0;
- strict: tool correct AND argument F1 == 1.0.
Aggregates report tool accuracy, strict accuracy, mean argument F1 conditional on the
correct tool (argument quality unpolluted by routing errors), and parse failures —
overall and per split ("pattern" vs "compositional")."""

from __future__ import annotations

import json
from pathlib import Path

from microlab.model.reference.chat_sft import END_SENTINEL
from microlab.model.reference.sft import format_chat

MARKER_TOOLS = ("clarify", "none")
DEMO_TOOL = "get_time"
# Tools every prompt includes: the few-shot demos use DEMO_TOOL and `clarify`.
ALWAYS_SHOWN = (DEMO_TOOL, *MARKER_TOOLS)

DIRECTIONS = (
    "You are a tool-calling assistant. Pick the single best tool for the request and "
    'reply with ONLY a JSON object: {"tool": "<name>", "arguments": {...}}. '
    'If a required argument is missing from the request, reply with the "clarify" tool. '
    'If no listed tool fits, reply with the "none" tool.\n\nTools:\n'
)

FEW_SHOT = [
    {
        "user": "What time is it right now in Tokyo?",
        "assistant": '{"tool": "get_time", "arguments": {"timezone": "Asia/Tokyo"}}',
    },
    {
        "user": "Set the thermostat.",
        "assistant": '{"tool": "clarify", "arguments": {}}',
    },
]


# ------------------------------------------------------------------------------ registry

def load_registry(path: Path) -> dict[str, dict]:
    """{tool name: tool} from registry.json, validated. Raises ValueError on structural
    problems — a malformed registry must stop the eval, not skew it."""
    tools = json.loads(Path(path).read_text(encoding="utf-8"))["tools"]
    out: dict[str, dict] = {}
    for t in tools:
        for key in ("name", "description", "params"):
            if key not in t:
                raise ValueError(f"tool {t.get('name')!r}: missing {key!r}")
        if t["name"] in out:
            raise ValueError(f"duplicate tool {t['name']!r}")
        for p in t["params"]:
            for key in ("name", "type", "required"):
                if key not in p:
                    raise ValueError(f"tool {t['name']!r} param {p.get('name')!r}: "
                                     f"missing {key!r}")
        out[t["name"]] = t
    for marker in MARKER_TOOLS:
        if marker not in out:
            raise ValueError(f"registry must define the {marker!r} marker tool")
    if DEMO_TOOL not in out:
        raise ValueError(f"registry must define the few-shot demo tool {DEMO_TOOL!r}")
    return out


def load_items(path: Path, registry: dict[str, dict]) -> list[dict]:
    """Eval items from items.jsonl, cross-checked against the registry: the expected
    tool exists and is shown, and expected arguments fit its schema."""
    items = [json.loads(line) for line in
             Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    seen_ids: set[str] = set()
    for it in items:
        for key in ("id", "split", "request", "tools", "expected"):
            if key not in it:
                raise ValueError(f"item {it.get('id')!r}: missing {key!r}")
        if it["id"] in seen_ids:
            raise ValueError(f"duplicate item id {it['id']!r}")
        seen_ids.add(it["id"])
        if it["split"] not in ("pattern", "compositional"):
            raise ValueError(f"item {it['id']!r}: bad split {it['split']!r}")
        exp = it["expected"]
        tool = registry.get(exp["tool"])
        if tool is None:
            raise ValueError(f"item {it['id']!r}: expected tool {exp['tool']!r} "
                             f"not in registry")
        shown = set(it["tools"]) | set(ALWAYS_SHOWN)
        for name in it["tools"]:
            if name not in registry:
                raise ValueError(f"item {it['id']!r}: shown tool {name!r} not in registry")
        if exp["tool"] not in shown:
            raise ValueError(f"item {it['id']!r}: expected tool not among shown tools")
        params = {p["name"]: p for p in tool["params"]}
        for arg in exp["arguments"]:
            if arg not in params:
                raise ValueError(f"item {it['id']!r}: expected arg {arg!r} not a param "
                                 f"of {exp['tool']!r}")
        for p in tool["params"]:
            if p["required"] and p["name"] not in exp["arguments"]:
                raise ValueError(f"item {it['id']!r}: required param {p['name']!r} of "
                                 f"{exp['tool']!r} missing from expected arguments")
    return items


# ------------------------------------------------------------------------------- prompts

def render_tool(tool: dict) -> str:
    """One compact line: name(required: type, optional?: type) — description."""
    sig = ", ".join(
        f"{p['name']}{'' if p['required'] else '?'}: {p['type']}" for p in tool["params"]
    )
    return f"- {tool['name']}({sig}) — {tool['description']}"


def shown_tools(item: dict, registry: dict[str, dict]) -> list[dict]:
    """The item's tools plus the always-present demo/marker tools, in registry order
    (deterministic regardless of item authoring order)."""
    names = set(item["tools"]) | set(ALWAYS_SHOWN)
    return [t for name, t in registry.items() if name in names]


def build_prompt(item: dict, registry: dict[str, dict]) -> str:
    """Full generation prompt: directions + tool list + request in the first user turn,
    then the two few-shot turns, then the item's request — all rendered with the trained
    multi-turn chat template (format_chat blocks separated by the '### End\\n' seam)."""
    tool_lines = "\n".join(render_tool(t) for t in shown_tools(item, registry))
    first_user = DIRECTIONS + tool_lines + "\n\nRequest: " + FEW_SHOT[0]["user"]
    parts = [format_chat(first_user)[0], FEW_SHOT[0]["assistant"] + END_SENTINEL + "\n"]
    parts += [format_chat("Request: " + FEW_SHOT[1]["user"])[0],
              FEW_SHOT[1]["assistant"] + END_SENTINEL + "\n"]
    parts.append(format_chat("Request: " + item["request"])[0])
    return "".join(parts)


# ------------------------------------------------------------------------------- scoring

def parse_call(reply: str) -> tuple[str, dict] | None:
    """(tool, arguments) from a model reply, or None when unparseable. The first JSON
    object in the reply is decoded with raw_decode (correct even with braces inside
    string values — a balanced-brace scan is not); a missing "arguments" key defaults
    to {} but a non-dict one is a parse failure."""
    decoder = json.JSONDecoder()
    pos = reply.find("{")
    while pos != -1:
        try:
            obj, _ = decoder.raw_decode(reply, pos)
        except json.JSONDecodeError:
            pos = reply.find("{", pos + 1)
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("tool"), str):
            return None
        args = obj.get("arguments", {})
        if not isinstance(args, dict):
            return None
        return obj["tool"], args
    return None


def _normalize(value):
    if isinstance(value, str):
        return value.strip().casefold()
    if isinstance(value, bool):
        return ("bool", value)  # tagged: True must not equal 1 (Python's True == 1)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in sorted(value.items())}
    return value


def argument_f1(expected: dict, predicted: dict) -> float:
    """F1 over argument key/value pairs; a pair matches when the key is present in both
    and the normalized values are equal. Both empty -> 1.0 (the empty call was right)."""
    if not expected and not predicted:
        return 1.0
    if not expected or not predicted:
        return 0.0
    matches = sum(
        1 for k, v in expected.items()
        if k in predicted and _normalize(predicted[k]) == _normalize(v)
    )
    precision = matches / len(predicted)
    recall = matches / len(expected)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_item(item: dict, reply: str) -> dict:
    """Per-item record: parse status, tool match, argument F1, strict correctness."""
    parsed = parse_call(reply)
    exp = item["expected"]
    if parsed is None:
        return {"id": item["id"], "split": item["split"], "parsed": False,
                "predicted": None, "tool_correct": False, "arg_f1": 0.0,
                "strict": False}
    tool, args = parsed
    tool_correct = tool == exp["tool"]
    f1 = argument_f1(exp["arguments"], args) if tool_correct else 0.0
    return {"id": item["id"], "split": item["split"], "parsed": True,
            "predicted": {"tool": tool, "arguments": args},
            "tool_correct": tool_correct, "arg_f1": f1,
            "strict": tool_correct and f1 == 1.0}


def aggregate(results: list[dict]) -> dict:
    """Overall + per-split metrics. arg_f1 is averaged over items where the tool was
    correct (argument quality, not routing); strict/tool accuracy over all items."""
    def block(rs: list[dict]) -> dict:
        n = len(rs)
        if n == 0:
            raise ValueError("no results to aggregate")
        correct_tool = [r for r in rs if r["tool_correct"]]
        return {
            "n": n,
            "tool_acc": len(correct_tool) / n,
            "strict_acc": sum(r["strict"] for r in rs) / n,
            "arg_f1_when_tool_correct": (
                sum(r["arg_f1"] for r in correct_tool) / len(correct_tool)
                if correct_tool else 0.0
            ),
            "parse_failures": sum(1 for r in rs if not r["parsed"]),
        }

    splits = sorted({r["split"] for r in results})
    return {"overall": block(results),
            "splits": {s: block([r for r in results if r["split"] == s]) for s in splits}}
