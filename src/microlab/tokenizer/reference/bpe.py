"""Reference byte-level BPE tokenizer (the oracle the owner diffs against).

Byte-level so decode(encode(text)) == text for arbitrary unicode. Training merges
the most-frequent adjacent token pair until the target vocab size is reached.
"""

from __future__ import annotations

from collections import Counter


def _get_stats(ids: list[int]) -> Counter[tuple[int, int]]:
    stats: Counter[tuple[int, int]] = Counter()
    for a, b in zip(ids, ids[1:], strict=False):
        stats[(a, b)] += 1
    return stats


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self) -> None:
        # merges: ordered map (pair) -> new_id; vocab: id -> bytes
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Train BPE merges on `text` until `vocab_size` tokens are reached."""
        assert vocab_size >= 256, "vocab_size must be >= 256 (byte base)"
        ids = list(text.encode("utf-8"))
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for new_id in range(256, vocab_size):
            stats = _get_stats(ids)
            if not stats:
                break
            # most frequent pair; ties broken by largest byte values (deterministic)
            best = max(stats, key=lambda p: (stats[p], -p[0], -p[1]))
            ids = _merge(ids, best, new_id)
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            if verbose:
                print(f"merge {new_id - 255}/{vocab_size - 256}: {best} -> {new_id}")

    def encode(self, text: str) -> list[int]:
        """Encode text to a list of token ids using learned merges."""
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = _get_stats(ids)
            # merge the pair with the lowest merge-id that is present (earliest learned)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back to a string."""
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")
