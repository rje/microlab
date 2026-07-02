"""Reference supervised fine-tuning tools (Phase 9): a chat template, SFT example building
with PROMPT LOSS MASKING (train only on response tokens), a padded collator, and masked
cross-entropy. The oracle the owner diffs their hand-written masking against.

The masking is the crux: labels at prompt positions are set to IGNORE_INDEX (-100) so the
loss ignores them and the model is trained only to produce the response."""

from __future__ import annotations

import torch
from torch.nn import functional as F

IGNORE_INDEX = -100

PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n{context}### Response:\n"
)


def format_chat(instruction: str, context: str = "", response: str = "") -> tuple[str, str]:
    """Return (prompt_text, response_text). Context is included as its own block when present.
    The response is what the model must learn to generate."""
    ctx = f"### Input:\n{context}\n\n" if context.strip() else ""
    prompt = PROMPT_TEMPLATE.format(instruction=instruction, context=ctx)
    return prompt, response


def build_sft_example(tok, prompt_text: str, response_text: str) -> tuple[list[int], list[int]]:
    """Tokenize prompt+response into input_ids, and labels where PROMPT positions are masked
    to IGNORE_INDEX so only the response contributes to the loss."""
    prompt_ids = tok.encode(prompt_text)
    response_ids = tok.encode(response_text)
    input_ids = prompt_ids + response_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + list(response_ids)
    return input_ids, labels


def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over (B, T, V) logits and (B, T) labels, ignoring IGNORE_INDEX positions
    (the prompt + padding). Applies the standard causal-LM shift: logits[:, :-1] predict
    labels[:, 1:], so position t of logits predicts token t+1."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


def collate_sft(
    examples: list[tuple[list[int], list[int]]], pad_id: int = 0, block_size: int | None = None
) -> dict[str, torch.Tensor]:
    """Pad a batch of (input_ids, labels) to the batch max length. input_ids pad with pad_id;
    labels pad with IGNORE_INDEX. Truncate to block_size if given."""
    if block_size is not None:
        examples = [(i[:block_size], la[:block_size]) for i, la in examples]
    maxlen = max(len(i) for i, _ in examples)
    inp, lab = [], []
    for i, la in examples:
        pad = maxlen - len(i)
        inp.append(i + [pad_id] * pad)
        lab.append(la + [IGNORE_INDEX] * pad)
    return {"input_ids": torch.tensor(inp, dtype=torch.long),
            "labels": torch.tensor(lab, dtype=torch.long)}


def train_sft(model, examples, train_cfg, pad_id: int = 0) -> dict:
    """Fine-tune `model` on SFT examples (list of (input_ids, labels)) using masked loss.
    Returns loss history."""
    from microlab.model.reference.train import _resolve_device

    torch.manual_seed(train_cfg.seed)
    device = _resolve_device(train_cfg.device)
    model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    model.train()
    rng = torch.Generator().manual_seed(train_cfg.seed)
    history = []
    for _ in range(train_cfg.steps):
        idx = torch.randint(len(examples), (train_cfg.batch_size,), generator=rng).tolist()
        batch = collate_sft([examples[i] for i in idx], pad_id, train_cfg.block_size)
        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)
        logits, _ = model(x)
        loss = masked_cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        history.append(loss.item())
    return {"final_loss": history[-1], "history": history, "device": device}
