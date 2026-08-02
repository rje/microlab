"""Fill-in-the-Middle (FIM) transform, PSM variant.

A left-to-right LM can only continue a prefix. Real code editing needs INFILLING: given
what comes before AND after a hole, write the middle. FIM (Bavarishi et al., 2022) buys
that for free at pretraining time by rewriting some documents so the middle is what gets
predicted last:

    prefix | middle | suffix   ->   <pre> prefix <suf> suffix <mid> middle

PSM (prefix-suffix-middle) rather than SPM: DeepSeek-Coder's published ablation puts PSM
ahead, and the 0.5 rate is theirs. Both papers agree the transform is applied at DOCUMENT
level, so the model still sees plenty of ordinary left-to-right text at the same rate.

The transform is its own inverse in the sense that matters: `defim` recovers the original
token sequence exactly, which is what the round-trip test asserts. If it did not, we would
be training on documents we cannot reason about.
"""

from __future__ import annotations

import numpy as np

FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"
FIM_TOKENS = (FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE)


class FIMConfig:
    """Sentinel ids resolved against a tokenizer, so the transform never guesses them."""

    def __init__(self, tokenizer) -> None:
        ids = [tokenizer.token_to_id(t) for t in FIM_TOKENS]
        if any(i is None for i in ids):
            missing = [t for t, i in zip(FIM_TOKENS, ids, strict=True) if i is None]
            raise ValueError(
                f"tokenizer lacks FIM sentinels {missing}. Build it with "
                f"data/tokenizers/code-49k-fim.json, not code-49k.json — silently skipping "
                f"FIM would produce a corpus that looks fine and teaches no infilling.")
        self.prefix, self.suffix, self.middle = ids


def fim_transform(doc: list[int] | np.ndarray, cfg: FIMConfig,
                  rng: np.random.Generator) -> list[int]:
    """Rewrite one document into PSM form. Two cut points, uniform, sorted.

    Documents shorter than 4 tokens are returned unchanged: there is no meaningful
    three-way split, and emitting an empty middle would train the model to answer
    infilling requests with nothing.
    """
    d = list(doc)
    if len(d) < 4:
        return d
    a, b = sorted(rng.integers(1, len(d), size=2).tolist())
    if a == b:                      # degenerate split -> empty middle; nudge it
        b = min(a + 1, len(d) - 1)
        if a >= b:
            return d
    prefix, middle, suffix = d[:a], d[a:b], d[b:]
    return ([cfg.prefix] + prefix + [cfg.suffix] + suffix + [cfg.middle] + middle)


def defim(doc: list[int], cfg: FIMConfig) -> list[int]:
    """Recover the original token order from a PSM document. Round-trip oracle."""
    if not doc or doc[0] != cfg.prefix:
        return list(doc)
    try:
        s = doc.index(cfg.suffix)
        m = doc.index(cfg.middle, s)
    except ValueError as e:
        raise ValueError(f"malformed FIM document: {e}") from e
    prefix, suffix, middle = doc[1:s], doc[s + 1:m], doc[m + 1:]
    return prefix + middle + suffix


def split_documents(tokens: np.ndarray, eot: int) -> list[np.ndarray]:
    """Split a packed token stream on EOT. Trailing partial document is kept."""
    cuts = np.flatnonzero(tokens == eot)
    out, start = [], 0
    for c in cuts:
        if c > start:
            out.append(tokens[start:c])
        start = c + 1
    if start < len(tokens):
        out.append(tokens[start:])
    return out
