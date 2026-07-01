"""Hand-write exercise (Phase 2): implement the core transformer pieces.

Fill in the four ``NotImplementedError`` bodies so ``tests/model/test_student.py``
passes. The tests diff your work against the reference oracle in
``microlab.model.reference`` by copying the reference's weights into your modules and
comparing outputs — so your math must match, not merely "look right". The submodule
names here mirror the reference exactly so ``load_state_dict`` transfers weights
cleanly. See ``docs/hand-write/phase2-gpt.md``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from microlab.model.reference.gpt import MLP, GPTConfig


class StudentCausalSelfAttention(nn.Module):
    """Multi-head causal self-attention. Same parameters/names as the reference
    (``c_attn``, ``c_proj``) so reference weights load straight in."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "implement causal multi-head self-attention: project x -> q,k,v; split into "
            "heads; scaled dot-product (scores / sqrt(head_dim)) with a causal mask; "
            "softmax; weight v; recombine heads; apply the output projection"
        )


class StudentBlock(nn.Module):
    """Pre-norm transformer block. Same submodule names as the reference block, and it
    reuses the reference ``MLP`` so only the attention + residual wiring is yours."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = StudentCausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "implement the pre-norm residual block: x = x + attn(ln_1(x)); "
            "x = x + mlp(ln_2(x))"
        )


def train_step(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer
) -> float:
    """One optimization step: zero grads, forward to the loss, backward, step. Return
    the scalar loss value (a float). The model's ``forward(x, y)`` returns
    ``(logits, loss)``."""
    raise NotImplementedError("implement the training step (zero_grad, forward, backward, step)")


@torch.no_grad()
def generate(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Autoregressive sampling loop. Crop the context to ``model.config.block_size``,
    take the last-step logits, and either argmax (``temperature == 0``) or sample from
    the (optionally top-k filtered, temperature-scaled) softmax. Append and repeat."""
    raise NotImplementedError("implement the autoregressive sampling loop")
