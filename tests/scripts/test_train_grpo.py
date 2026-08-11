"""scripts/train_grpo.py pure pieces: the RM scoring-sequence construction (pinned to the
EXACT training-time constructor in scripts/train_reward_model.py — a silent mismatch would
make every reward wrong), the prompt-pool block guard, and the sentinel/pad consistency the
loss and the reward construction both rely on. Loaded via importlib since scripts/ isn't a
package."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from microlab.model.reference.sft import format_chat
from microlab.tokenizer.fast import FastTokenizer
from microlab.train import grpo

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tg = _load("train_grpo")
trm = _load("train_reward_model")


class _ByteTok:
    """Byte-per-char tokenizer so length arithmetic in the pool guard is exact."""

    def encode(self, s: str) -> list[int]:
        return list(s.encode("utf-8"))


# ---------------------------------------------------------------- RM scoring construction


def test_scoring_sequences_match_training_construction_exactly():
    tok = FastTokenizer.train(["what is the capital of France? Paris, obviously."] * 4,
                              vocab_size=300)
    prompt, _ = format_chat("what is the capital?")
    texts = ["Paris", "the capital is Paris, obviously", ""]
    seqs = tg.build_scoring_sequences(tok, prompt, texts, block_size=1024)
    assert len(seqs) == 3
    for text, seq in zip(texts, seqs, strict=True):
        # Pin against the training-time constructor itself...
        pairs, _ = trm.build_reward_sequences(
            tok, [{"prompt": prompt, "chosen": text, "rejected": text}], 1024)
        assert seq == pairs[0][0]
        # ...and against the explicit construction, so a drift in EITHER is caught loudly.
        assert seq == tok.encode(prompt) + tok.encode(text + trm.END_SENTINEL)


def test_scoring_sequences_raise_on_block_overflow():
    # Training SKIPS an overlong pair; here a skip would silently misalign rewards with
    # rollouts, so it must raise instead.
    tok = _ByteTok()
    with pytest.raises(ValueError, match="block"):
        tg.build_scoring_sequences(tok, "p: ", ["x" * 50], block_size=16)


# ---------------------------------------------------------------- prompt pool


def test_build_prompt_pool_templates_and_guards():
    rows = [{"instruction": "short", "context": ""},
            {"instruction": "with ctx", "context": "some context"},
            {"instruction": "x" * 500, "context": ""}]  # too long for the block
    pool, skipped = tg.build_prompt_pool(_ByteTok(), rows, max_new=16, block_size=128)
    assert skipped == 1
    assert pool == [format_chat("short", "")[0], format_chat("with ctx", "some context")[0]]


def test_build_prompt_pool_guard_reserves_room_for_the_sentinel():
    # A prompt where prompt + max_new fits but prompt + max_new + sentinel does NOT must be
    # skipped: the trained sequence appends END_SENTINEL beyond the sampled tokens.
    tok = _ByteTok()
    prompt = format_chat("abcdefghijkl", "")[0]
    fits = len(tok.encode(prompt)) + 16  # room for prompt + max_new but NOT the sentinel
    rows = [{"instruction": "abcdefghijkl", "context": ""},
            {"instruction": "ab", "context": ""}]
    pool, skipped = tg.build_prompt_pool(tok, rows, max_new=16, block_size=fits)
    assert pool == [format_chat("ab", "")[0]] and skipped == 1


def test_build_prompt_pool_raises_when_empty():
    with pytest.raises(ValueError, match="no usable prompts"):
        tg.build_prompt_pool(_ByteTok(), [{"instruction": "y" * 500, "context": ""}],
                             max_new=16, block_size=128)


# ---------------------------------------------------------------- cross-file invariants


def test_sentinel_and_pad_match_reward_training_and_library():
    # train_grpo raises at import if these drift; assert them directly too so the invariant
    # is stated where a future reader looks for it.
    assert trm.END_SENTINEL == grpo.END_SENTINEL
    assert trm.PAD_ID == grpo.PAD_ID


# ---------------------------------------------------------------- executor oracle


def test_build_executor_oracle_maps_prompts_to_io():
    pool = [{"instruction": "double n", "io": [{"input": "2\n", "output": "4\n"}]},
            {"instruction": "x" * 5000, "io": [{"input": "1\n", "output": "1\n"}]}]  # too long
    prompts, score = tg.build_executor_oracle(_ByteTok(), pool, max_new=64,
                                              block_size=2048, timeout_s=5.0)
    assert len(prompts) == 1                       # oversized row skipped (and counted)
    want_prompt, _ = format_chat("double n", "")
    assert prompts[0] == want_prompt
    got = score(prompts[0], ["```python\nn=int(input());print(n*2)\n```"])
    assert got == [1.0]


def test_build_executor_oracle_raises_when_empty():
    with pytest.raises(ValueError, match="no usable pool rows"):
        tg.build_executor_oracle(_ByteTok(), [{"instruction": "y" * 5000,
                                               "io": [{"input": "1\n", "output": "1\n"}]}],
                                 max_new=16, block_size=128, timeout_s=5.0)
