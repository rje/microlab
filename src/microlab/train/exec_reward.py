"""Executor-backed GRPO reward: rollout text -> extracted code -> sandbox -> reward.

The reward is the FRACTION of a problem's checked I/O cases the extracted solution passes
(dense — a binary all-pass reward zeroes the group advantage on most groups at ~14% pass
rates). This is the `score_texts` oracle run_grpo injects; the RM oracle in train_grpo.py is
the precedent. The executor is ground truth (build capability, don't distill).
"""
from __future__ import annotations

import torch as _torch

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


def signal_bearing(successes: int, k: int) -> bool:
    """Only mixed groups carry GRPO advantage: all-fail and all-pass both standardize to
    zero. The pre-pass keeps problems where the policy sometimes-but-not-always succeeds."""
    return 0 < successes < k


def sample_solutions(model, tok, prompt: str, k: int, *, max_new: int = 300,
                     temp: float = 0.8, top_k: int | None = 40, seed: int = 0,
                     device: str = "cuda") -> list[str]:
    """k sampled replies for one chat-formatted prompt, generated as ONE k-wide batch
    (a rerun with the same (prompt, k, seed) reproduces the same sample set). Returns raw
    reply texts (caller extracts/rewards).

    Args:
        model: The language model to sample from.
        tok: Raw tokenizers.Tokenizer object with .encode(text).ids and .decode(ids).
        prompt: Chat-formatted prompt string.
        k: Number of samples to generate.
        max_new: Maximum tokens to generate.
        temp: Sampling temperature.
        top_k: Top-k for sampling.
        seed: Seed for the batch generator (reproducible at the (prompt, k, seed) level).
        device: Device to generate on.

    Returns:
        List of k sampled reply texts (not including the prompt).
    """
    from microlab.infer.reference.kv_cache import generate_cached
    # One BATCHED generate call: the prompt stacked k-wide. ~8x faster than k sequential
    # calls (generation dominates the pre-pass/self-gen wall clock; the sequential version
    # projected ~23h over the full pool). Reproducibility contract: one generator seeded
    # with `seed` for the whole batch — a rerun with the same (prompt, k, seed) reproduces
    # the same k samples as a set (not per-sample seeds as the sequential version had).
    prompt_ids = tok.encode(prompt).ids
    ids = _torch.tensor([prompt_ids], device=device).expand(k, -1).contiguous()
    gen = _torch.Generator(device=device).manual_seed(seed)
    with _torch.no_grad():
        out = generate_cached(model, ids, max_new, temperature=temp, top_k=top_k,
                              generator=gen)
    return [tok.decode(out[i].tolist())[len(prompt):] for i in range(k)]
