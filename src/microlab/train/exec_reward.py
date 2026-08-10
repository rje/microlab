"""Executor-backed GRPO reward: rollout text -> extracted code -> sandbox -> reward.

The reward is the FRACTION of a problem's checked I/O cases the extracted solution passes
(dense — a binary all-pass reward zeroes the group advantage on most groups at ~14% pass
rates). This is the `score_texts` oracle run_grpo injects; the RM oracle in train_grpo.py is
the precedent. The executor is ground truth (build capability, don't distill).
"""
from __future__ import annotations

from microlab.data.code_sft import assemble_io_program
from microlab.evals.code.executor import run_python
from microlab.evals.code.prompts import extract_code_block


def extract_solution(reply: str) -> str:
    """Code from a chat rollout: sentinel-truncated, unfenced (reuses the eval extractor —
    training and eval must extract identically or reward would diverge from measurement)."""
    return extract_code_block(reply)


def io_reward(solution: str, io_cases: list[dict], timeout_s: float = 5.0) -> float:
    """Fraction of io_cases the solution passes. Empty/whitespace solution -> 0.0."""
    if not solution.strip() or not io_cases:
        return 0.0
    passed = 0
    for c in io_cases:
        prog = assemble_io_program(solution, c["input"], c["output"])
        if run_python(prog, timeout_s=timeout_s).passed:
            passed += 1
    return passed / len(io_cases)


def make_exec_score_texts(io_by_prompt: dict[str, list[dict]], timeout_s: float = 5.0):
    """score_texts(prompt, texts) -> rewards, keyed by the exact chat-formatted prompt.
    An unknown prompt raises KeyError — a pool/loop wiring bug must fail loudly, not train
    on silent zero rewards."""
    def score_texts(prompt: str, texts: list[str]) -> list[float]:
        cases = io_by_prompt[prompt]
        return [io_reward(extract_solution(t), cases, timeout_s=timeout_s) for t in texts]
    return score_texts
