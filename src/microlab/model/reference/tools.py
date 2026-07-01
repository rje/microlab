"""Reference tool-use / agent tools (Phase 12). Parse a tool call from a model's text
output, validate it against a tool schema, measure schema validity, and run a minimal
ReAct-style loop (think -> tool call -> observation -> ... -> final answer). The oracle
the owner diffs their hand-written parser/validator against."""

from __future__ import annotations

import json
import re
from collections.abc import Callable

_TOOL_RE = re.compile(r"<tool>\s*(\{.*?\})\s*</tool>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


def parse_tool_call(text: str) -> dict | None:
    """Extract a tool call: a JSON object {"name": ..., "arguments": {...}} inside
    <tool>...</tool>. Returns the dict, or None if absent/unparseable."""
    m = _TOOL_RE.search(text)
    if not m:
        return None
    try:
        call = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(call, dict) or "name" not in call:
        return None
    call.setdefault("arguments", {})
    return call


def parse_final_answer(text: str) -> str | None:
    """Extract a final answer inside <answer>...</answer>, or None."""
    m = _ANSWER_RE.search(text)
    return m.group(1).strip() if m else None


def validate_tool_call(call: dict, schema: dict) -> bool:
    """Validate a parsed call against a schema of the form
    {tool_name: {"required": [arg, ...]}}. The tool must exist and all required args be
    present in call["arguments"]."""
    if not isinstance(call, dict) or call.get("name") not in schema:
        return False
    required = schema[call["name"]].get("required", [])
    args = call.get("arguments", {})
    return isinstance(args, dict) and all(r in args for r in required)


def schema_validity_rate(outputs: list[str], schema: dict) -> float:
    """Fraction of outputs that contain a parseable AND schema-valid tool call."""
    if not outputs:
        return 0.0
    def _valid(o: str) -> bool:
        call = parse_tool_call(o)
        return call is not None and validate_tool_call(call, schema)

    ok = sum(1 for o in outputs if _valid(o))
    return ok / len(outputs)


def run_tool_loop(
    model: Callable[[str], str], tools: dict[str, Callable[[dict], str]], schema: dict,
    prompt: str, max_steps: int = 5,
) -> dict:
    """Minimal ReAct loop. `model(context) -> text` emits either a <tool> call or an
    <answer>. Valid tool calls are executed (observation appended); invalid ones append an
    error. Stops on a final answer or max_steps. Returns {answer, steps, transcript}."""
    context = prompt
    transcript = []
    for step in range(max_steps):
        out = model(context)
        transcript.append(out)
        answer = parse_final_answer(out)
        if answer is not None:
            return {"answer": answer, "steps": step + 1, "transcript": transcript}
        call = parse_tool_call(out)
        if call is not None and validate_tool_call(call, schema):
            obs = tools[call["name"]](call.get("arguments", {}))
        else:
            obs = "error: invalid tool call"
        context = context + "\n" + out + "\n<observation>" + str(obs) + "</observation>"
    return {"answer": None, "steps": max_steps, "transcript": transcript}
