"""Hand-write exercise (Phase 15): tool-call parsing, schema validation, validity rate, and
the ReAct tool loop.

Fill in the ``NotImplementedError`` bodies so ``tests/exercises/test_phase15_tools.py`` passes.
Graded against ``microlab.model.reference.tools``. See docs/hand-write/phase15-tools.md.
"""

from __future__ import annotations

from collections.abc import Callable


def parse_tool_call(text: str) -> dict | None:
    """Extract a tool call: a JSON object {"name": ..., "arguments": {...}} inside
    <tool>...</tool>. Return the dict only if it parses AND is a dict AND carries a "name"
    (default a missing "arguments" to {}); return None on any failure — no tag, bad JSON, or
    no name. See docs/hand-write/phase15-tools.md.
    """
    raise NotImplementedError(
        "find and JSON-parse the <tool> body (it may be pretty-printed across lines), then "
        "gate on the dict-with-a-name contract above"
    )


def validate_tool_call(call: dict, schema: dict) -> bool:
    """Validate a parsed call against a schema of the form
    {tool_name: {"required": [arg, ...]}}: the tool name must exist in the schema AND every
    required arg must be present in call["arguments"]. See docs/hand-write/phase15-tools.md.
    """
    raise NotImplementedError(
        "check the call's tool name is known and all its schema-required arguments are present"
    )


def schema_validity_rate(outputs: list[str], schema: dict) -> float:
    """Fraction of outputs that contain a parseable AND schema-valid tool call (0.0 for an
    empty list) — the training/eval signal for "is the model speaking the tool interface".
    See docs/hand-write/phase15-tools.md.
    """
    raise NotImplementedError(
        "fraction of outputs that both parse and validate; guard the empty-list case"
    )


def run_tool_loop(
    model: Callable[[str], str], tools: dict[str, Callable[[dict], str]], schema: dict,
    prompt: str, max_steps: int = 5,
) -> dict:
    """Minimal ReAct loop (think -> act -> observe -> ...). Each step calls
    ``model(context) -> text``; the context accumulates across steps so the model conditions
    on the growing transcript.

    Contract (graded against the reference):
    - If the step's text carries a final answer, stop and return it. (Reuse
      ``parse_final_answer`` from ``microlab.model.reference.tools`` — it detects
      <answer>...</answer>.)
    - Otherwise parse a tool call: if it parses AND validates against ``schema``, execute
      ``tools[name](arguments)`` and inject the result as an observation into the context;
      if it doesn't, inject a fixed error-feedback observation instead so the model can retry.
    - Append the model's text (and the observation) to the context each step.
    - Terminate on the first final answer, or after ``max_steps`` model calls with no answer.
    - Return a dict with keys: ``answer`` (the string, or None if it never answered),
      ``steps`` (how many model calls were made), and ``transcript`` (the list of raw model
      outputs, one per step).

    (ReAct: Yao et al., Reasoning + Acting. See docs/hand-write/phase15-tools.md.)
    """
    raise NotImplementedError(
        "drive the think/act/observe loop: call the model, return on a final answer, else "
        "execute-or-error the tool call and grow the context; return the answer/steps/"
        "transcript dict per the contract above"
    )
