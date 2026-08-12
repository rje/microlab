"""Pure candidate-selection rules for the test-time harness.

Every function consumes ALREADY-COMPUTED results (strings, signatures, counts) — no
sandbox calls, no IO — so the selection science is hermetically testable. The impure
execution that produces behavioral signatures lives in microlab.infer.behavior.
"""
from __future__ import annotations

import random
import re

_WS = re.compile(r"\s+")


def normalize_code(s: str) -> str:
    """Whitespace-collapsed form for text-plurality (tabs/spaces/newlines equivalent)."""
    return _WS.sub(" ", s.strip())


def first_sample() -> int:
    """The floor selector: always the first draw (an unbiased single sample)."""
    return 0


def text_plurality(candidates: list[str]) -> int:
    """Index of the first candidate whose NORMALIZED text is most frequent. A floor
    baseline only — semantically equal code has many textual forms (see spec I4)."""
    if not candidates:
        raise ValueError("text_plurality requires at least one candidate")
    norm = [normalize_code(c) for c in candidates]
    counts: dict[str, int] = {}
    for n in norm:
        counts[n] = counts.get(n, 0) + 1
    best = max(counts.values())
    for i, n in enumerate(norm):
        if counts[n] == best:
            return i
    raise AssertionError("unreachable: candidates nonempty")


def behavior_clusters(signatures: list[tuple]) -> list[list[int]]:
    """Group candidate indices by identical behavioral signature. Largest cluster first;
    ties broken by smallest first-index (deterministic)."""
    groups: dict[tuple, list[int]] = {}
    for i, s in enumerate(signatures):
        groups.setdefault(s, []).append(i)
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


def pick_from_cluster(cluster: list[int], candidates: list[str], rule: str = "shortest",
                      seed: int = 0) -> int:
    """Pick one index from a cluster. rule='shortest' (Occam tiebreak — ablated, can favor
    degenerate code) or 'random' (seeded). Unknown rule raises."""
    if rule == "shortest":
        return min(cluster, key=lambda i: (len(candidates[i]), i))
    if rule == "random":
        return random.Random(seed).choice(sorted(cluster))
    raise ValueError(f"unknown pick rule {rule!r}")


def select_by_self_tests(assert_pass_counts: list[int]) -> int:
    """Index with the most self-test asserts passed; first index wins ties."""
    best = max(assert_pass_counts)
    return assert_pass_counts.index(best)
