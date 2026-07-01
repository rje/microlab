"""Story-generation eval (TinyStories-style): the model completes story beginnings, and a
judge scores each completion on a rubric. Perplexity measures modeling; this measures
whether the generations are actually good stories. The judge is injectable — a mock in
tests, a local LLM (Ollama) in practice."""

from __future__ import annotations

import json
from collections.abc import Callable

import torch

RUBRIC = ["grammar", "coherence", "creativity", "consistency"]

DEFAULT_PROMPTS = [
    "Once upon a time, there was a little girl who",
    "One day, a boy named Tom found a shiny",
    "The cat and the dog were best friends. One morning they",
    "Lily wanted to bake a cake for her mom, so she",
    "In a big green forest, a small rabbit was looking for",
]


def complete_stories(model, tokenizer, prompts=None, max_new_tokens: int = 120,
                     temperature: float = 0.7, top_k: int | None = 40,
                     device: str = "cpu") -> list[dict]:
    """Generate a continuation for each prompt. Returns [{prompt, completion}, ...]."""
    from microlab.model.reference.sample import generate as _gen

    prompts = prompts if prompts is not None else DEFAULT_PROMPTS
    model.to(device)
    out = []
    for p in prompts:
        ids = tokenizer.encode(p)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        gen = _gen(model, idx, max_new_tokens, temperature, top_k)
        out.append({"prompt": p, "completion": tokenizer.decode(gen[0].tolist()[len(ids):])})
    return out


def parse_scores(text: str) -> dict[str, int]:
    """Parse a judge's JSON rubric scores; missing -> 3, clamp to 1..5."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        data = {}
    return {r: max(1, min(5, int(data.get(r, 3)))) for r in RUBRIC}


def run_story_eval(completions: list[dict],
                   judge_fn: Callable[[str, str], dict[str, int]]) -> dict:
    """Judge each completion (judge_fn(prompt, completion) -> {rubric: score}); return
    per-rubric means + the scored samples."""
    scored = [{**c, "scores": judge_fn(c["prompt"], c["completion"])} for c in completions]
    means = {}
    if scored:
        for r in RUBRIC:
            means[r] = sum(s["scores"].get(r, 0) for s in scored) / len(scored)
        means["overall"] = sum(means[r] for r in RUBRIC) / len(RUBRIC)
    return {"means": means, "samples": scored}


def ollama_judge(model: str = "qwen2.5:7b",
                 host: str = "http://localhost:11434") -> Callable[[str, str], dict[str, int]]:
    """A judge_fn backed by a local Ollama model. Asks it to score 1-5 on the rubric and
    returns parsed scores. (Not unit-tested — needs a running Ollama.)"""
    import requests

    def judge(prompt: str, completion: str) -> dict[str, int]:
        ask = (
            "You are grading a short children's story completion. Score each 1-5 "
            f"({', '.join(RUBRIC)}). Story beginning:\n{prompt}\nCompletion:\n{completion}\n"
            'Reply with ONLY JSON like {"grammar":4,"coherence":3,"creativity":4,"consistency":3}.'
        )
        resp = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": ask, "stream": False, "format": "json"},
            timeout=120,
        )
        return parse_scores(resp.json().get("response", "{}"))

    return judge
