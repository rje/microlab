"""Reference architecture variants (Phase 3): RMSNorm, rotary position embeddings
(RoPE), and a SwiGLU MLP, plus a flag-configurable GPT and helpers to build ablations.
These are the known-correct versions the owner diffs hand-written variants against."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from microlab.model.reference.gpt import MLP, CausalSelfAttention, GPTConfig


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean-subtraction, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0,
                     device=None, dtype=torch.float32):
    """Precompute cos/sin tables, shape (seq_len, head_dim//2)."""
    assert head_dim % 2 == 0
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, theta)  # (seq_len, head_dim//2)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x of shape (B, n_head, T, head_dim). cos/sin: (T, head_dim//2)."""
    T = x.size(-2)
    cos = torch.cat((cos, cos), dim=-1)[:T][None, None, :, :]
    sin = torch.cat((sin, sin), dim=-1)[:T][None, None, :, :]
    return x * cos + _rotate_half(x) * sin


class RoPECausalSelfAttention(nn.Module):
    """Causal self-attention with rotary position embeddings on q,k (no learned wpe)."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        head_dim = config.n_embd // config.n_head
        cos, sin = build_rope_cache(config.block_size, head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, kv_cache=None) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        offset = kv_cache[0].seq_len if kv_cache is not None else 0
        cos = self.rope_cos[offset:offset + T].to(q.dtype)
        sin = self.rope_sin[offset:offset + T].to(q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if kv_cache is not None:
            cache, layer = kv_cache
            k, v = cache.append(layer, k, v)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=(q.size(-2) == k.size(-2)),
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class GQAAttention(nn.Module):
    """Grouped-query attention with RoPE: n_head query heads share n_kv_head K/V heads
    (n_kv_head == 1 is multi-query attention). Halves-to-quarters the KV projection —
    and, later, the KV cache — at near-zero quality cost (Ainslie et al., 2023)."""

    def __init__(self, config: VariantConfig) -> None:
        super().__init__()
        assert config.pos == "rope", "GQAAttention is built for the RoPE block"
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.kv_proj = nn.Linear(
            config.n_embd, 2 * config.n_kv_head * self.head_dim, bias=config.bias
        )
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        cos, sin = build_rope_cache(config.block_size, self.head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, kv_cache=None) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k, v = self.kv_proj(x).split(self.n_kv_head * self.head_dim, dim=2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        offset = kv_cache[0].seq_len if kv_cache is not None else 0
        cos = self.rope_cos[offset:offset + T].to(q.dtype)
        sin = self.rope_sin[offset:offset + T].to(q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if kv_cache is not None:
            cache, layer = kv_cache
            k, v = cache.append(layer, k, v)
        groups = self.n_head // self.n_kv_head
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=(q.size(-2) == k.size(-2)),
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward (GLU variant): w2(silu(w1 x) * w3 x). Hidden dim ~ 8/3 * n_embd
    (rounded to a multiple of 8) so param count is comparable to the 4x GELU MLP."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        hidden = int(8 / 3 * config.n_embd)
        hidden = ((hidden + 7) // 8) * 8
        self.w1 = nn.Linear(config.n_embd, hidden, bias=False)
        self.w3 = nn.Linear(config.n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


@dataclass
class VariantConfig(GPTConfig):
    norm: str = "layer"   # "layer" | "rms"
    pos: str = "learned"  # "learned" | "rope"
    mlp: str = "gelu"     # "gelu" | "swiglu"
    # None -> classic multi-head attention (fused c_attn), bit-identical to before this
    # field existed. Set to a divisor of n_head for grouped-query attention (1 == MQA).
    n_kv_head: int | None = None


def _make_norm(kind: str, dim: int) -> nn.Module:
    return RMSNorm(dim) if kind == "rms" else nn.LayerNorm(dim)


class VariantBlock(nn.Module):
    def __init__(self, config: VariantConfig) -> None:
        super().__init__()
        self.ln_1 = _make_norm(config.norm, config.n_embd)
        if getattr(config, "n_kv_head", None) is not None:
            self.attn = GQAAttention(config)
        elif config.pos == "rope":
            self.attn = RoPECausalSelfAttention(config)
        else:
            self.attn = CausalSelfAttention(config)
        self.ln_2 = _make_norm(config.norm, config.n_embd)
        self.mlp = SwiGLUMLP(config) if config.mlp == "swiglu" else MLP(config)

    def forward(self, x: torch.Tensor, kv_cache=None) -> torch.Tensor:
        if kv_cache is not None:
            x = x + self.attn(self.ln_1(x), kv_cache=kv_cache)
        else:
            x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class VariantGPT(nn.Module):
    """A GPT whose norm / positional / MLP choices are set by flags. With all defaults
    ('layer','learned','gelu') it's architecturally the Phase-2 baseline."""

    def __init__(self, config: VariantConfig) -> None:
        super().__init__()
        self.config = config
        # Trainer flips this on to trade compute for ~30x less activation memory; only the
        # training forward path (kv_cache is None) is ever checkpointed.
        self.grad_checkpoint = False
        modules = dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([VariantBlock(config) for _ in range(config.n_layer)]),
            ln_f=_make_norm(config.norm, config.n_embd),
        )
        if config.pos == "learned":
            modules["wpe"] = nn.Embedding(config.block_size, config.n_embd)
        self.transformer = nn.ModuleDict(modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, kv_cache=None):
        _, T = idx.shape
        assert T <= self.config.block_size, f"sequence length {T} > block_size"
        if kv_cache is not None:
            assert self.config.pos == "rope", "KV cache requires the RoPE block"
        x = self.transformer.wte(idx)
        if self.config.pos == "learned":
            pos = torch.arange(T, device=idx.device)
            x = x + self.transformer.wpe(pos)
        x = self.transformer.drop(x)
        for i, block in enumerate(self.transformer.h):
            if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                # Training path only: kv_cache is always None here (caching asserts
                # pos=="rope" and runs under no_grad in eval/generation).
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x, kv_cache=(kv_cache, i) if kv_cache is not None else None)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
