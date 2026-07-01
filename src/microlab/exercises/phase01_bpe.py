"""Hand-write exercise: a byte-level BPE tokenizer.

Implement `train` / `encode` / `decode` so `tests/tokenizer/test_bpe.py` passes. The
last test diffs your tokenizer against the reference oracle at
`microlab.tokenizer.reference.bpe` — so your merges must match it exactly, which means
matching its **deterministic tie-break**: when two adjacent pairs are equally frequent,
merge the one with the larger byte values (`(count, -a, -b)` argmax). Try it yourself
before reading the reference. See docs/hand-write/phase1-bpe.md.
"""

from __future__ import annotations


class BPETokenizer:
    def __init__(self) -> None:
        # merges: (pair) -> new_id, in learned order; vocab: id -> bytes
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        raise NotImplementedError(
            "implement BPE training — merge the most-frequent adjacent pair until "
            "len(vocab) == vocab_size; see docs/hand-write/phase1-bpe.md"
        )

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError("implement BPE encode (apply merges in learned order)")

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError("implement BPE decode (ids -> bytes -> utf-8)")
