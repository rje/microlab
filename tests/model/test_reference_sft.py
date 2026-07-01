import pytest
import torch

from microlab.model.reference.gpt import GPT, GPTConfig
from microlab.model.reference.sft import (
    IGNORE_INDEX,
    build_sft_example,
    collate_sft,
    format_chat,
    masked_cross_entropy,
    train_sft,
)
from microlab.model.reference.train import TrainConfig


class _FakeTok:
    """Char-code tokenizer for deterministic tests (encode = list of byte values)."""

    def encode(self, s):
        return list(s.encode("utf-8"))


def test_format_chat_includes_parts():
    prompt, resp = format_chat("Sum 2+2", context="", response="4")
    assert "Sum 2+2" in prompt and "### Response:" in prompt and resp == "4"
    p2, _ = format_chat("q", context="some ctx", response="a")
    assert "### Input:" in p2 and "some ctx" in p2


def test_build_sft_example_masks_prompt():
    tok = _FakeTok()
    prompt, resp = "PROMPT>", "RESP"
    input_ids, labels = build_sft_example(tok, prompt, resp)
    n_prompt = len(tok.encode(prompt))
    assert input_ids == tok.encode(prompt) + tok.encode(resp)
    assert labels[:n_prompt] == [IGNORE_INDEX] * n_prompt          # prompt masked
    assert labels[n_prompt:] == tok.encode(resp)                   # response supervised
    assert len(labels) == len(input_ids)


def test_masked_cross_entropy_ignores_prompt_positions():
    torch.manual_seed(0)
    logits = torch.randn(1, 5, 16)
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 3, 7, 1]])
    loss = masked_cross_entropy(logits, labels)
    # only the 3 non-masked positions matter: changing a masked label must not change loss
    labels2 = labels.clone()
    labels2[0, 0] = 9  # still... it's IGNORE_INDEX, change a masked slot
    labels2[0, 1] = IGNORE_INDEX
    assert torch.allclose(loss, masked_cross_entropy(logits, labels2))


def test_masked_cross_entropy_all_masked_is_nan_or_zero_safe():
    # a fully-masked row: F.cross_entropy returns nan; assert it doesn't raise
    logits = torch.randn(1, 3, 8)
    labels = torch.full((1, 3), IGNORE_INDEX)
    _ = masked_cross_entropy(logits, labels)  # should not raise


def test_collate_pads_inputs_and_labels():
    batch = collate_sft([([1, 2, 3], [IGNORE_INDEX, 2, 3]), ([4, 5], [IGNORE_INDEX, 5])], pad_id=0)
    assert batch["input_ids"].shape == (2, 3) and batch["labels"].shape == (2, 3)
    assert batch["input_ids"][1].tolist() == [4, 5, 0]           # pad_id
    assert batch["labels"][1].tolist() == [IGNORE_INDEX, 5, IGNORE_INDEX]  # pad -> ignore


def test_collate_truncates_to_block_size():
    batch = collate_sft([([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])], pad_id=0, block_size=3)
    assert batch["input_ids"].shape == (1, 3)


@pytest.mark.gpu
def test_train_sft_reduces_masked_loss_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    torch.manual_seed(0)
    tok = _FakeTok()
    # a few tiny examples over a small byte vocab
    examples = [build_sft_example(tok, f"Q{i}:", f" A{i}") for i in range(8)]
    m = GPT(GPTConfig(vocab_size=256, block_size=64, n_layer=2, n_head=2, n_embd=64))
    stats = train_sft(
        m, examples, TrainConfig(steps=100, batch_size=4, block_size=64, device="cuda")
    )
    assert stats["device"] == "cuda"
    assert stats["history"][-1] < stats["history"][0]  # SFT loss drops
