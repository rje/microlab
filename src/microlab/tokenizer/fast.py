"""Fast production tokenizer (Phase-1 real-scale): a byte-level BPE trained with the
HuggingFace `tokenizers` Rust library (tokenizes GB in seconds) — the scale replacement
for the hand-written reference BPE (same encode/decode interface). 32k vocab is well-matched
to a 150M-1B model."""

from __future__ import annotations

from collections.abc import Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOT = "<|endoftext|>"


class FastTokenizer:
    """Byte-level BPE with the same encode/decode surface as the reference BPE."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tok = tokenizer

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int = 32000,
              save_path: str | None = None) -> FastTokenizer:
        tok = Tokenizer(models.BPE(unk_token=None))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, special_tokens=[EOT],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tok.train_from_iterator(texts, trainer=trainer)
        if save_path:
            tok.save(save_path)
        return cls(tok)

    @classmethod
    def load(cls, path: str) -> FastTokenizer:
        return cls(Tokenizer.from_file(path))

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def eot_token(self) -> int:
        return self._tok.token_to_id(EOT)
