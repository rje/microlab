"""Prompt construction + completion extraction for the two checkpoint families.

BASE mode: the model is a raw LM. HumanEval's prompt (signature + docstring) is fed
verbatim and the model writes the body; the solution is prompt + completion, truncated at
the first token sequence that starts a new top-level construct. MBPP has no signature
prefix, so the prompt is the canonical docstring form (description + first assert) and
the completion IS the solution — which is why "\\ndef " must NOT be a stop there (the
model's own `def` opens the answer) while for HumanEval it must.

CHAT mode: the instruction is wrapped in the SFT template (format_chat — single source of
truth), generation stops at the trained sentinels, and the reply is mined for a fenced
code block (falling back to the raw reply only when no fence exists — small chat models
often skip the fence entirely, and the raw text is then the honest candidate)."""

from __future__ import annotations

from microlab.evals.code.tasks import CodeTask
from microlab.model.reference.chat_sft import END_SENTINEL
from microlab.model.reference.sft import format_chat

CHAT_STOPS = [END_SENTINEL, "### End", "\n### Instruction:"]

# Base-mode stops: any of these opening at top level means the solution is finished.
BASE_STOPS = {
    "humaneval": ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", '\n"""', "\n#", "\n@"],
    "mbpp": ["\nassert ", "\nclass ", "\nif __name__", "\nprint(", '\n"""', "\n#"],
}


def base_prompt(task: CodeTask) -> str:
    return task.prompt


def chat_prompt(task_instruction: str) -> str:
    return format_chat(task_instruction)[0]


def truncate_at(text: str, stops: list[str]) -> str:
    cut = min((text.find(s) for s in stops if s in text), default=-1)
    return text[:cut] if cut >= 0 else text


def base_solution(dataset: str, task: CodeTask, completion: str) -> str:
    """Assemble the candidate solution from a base-mode completion."""
    body = truncate_at(completion, BASE_STOPS[dataset])
    if dataset == "humaneval":
        return task.prompt + body
    return body


def extract_code_block(reply: str) -> str:
    """First fenced code block of a chat reply (optional language tag), else the raw
    reply stripped of the stop sentinels."""
    reply = truncate_at(reply, CHAT_STOPS)
    parts = reply.split("```")
    if len(parts) >= 2:  # an unclosed fence (cut off by max_new) still yields its block
        block = parts[1]
        first_nl = block.find("\n")
        if first_nl != -1 and block[:first_nl].strip().isalpha():
            block = block[first_nl + 1:]  # drop the ```python tag line
        return block.strip("\n")
    return reply.strip()


def chat_solution(reply: str) -> str:
    return extract_code_block(reply)
