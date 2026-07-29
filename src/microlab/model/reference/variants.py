"""Reference architecture variants (Phase 3): RMSNorm, rotary position embeddings
(RoPE), a SwiGLU MLP, and the Peri-LN block layout, plus a flag-configurable GPT and
helpers to build ablations. This file also carries grouped-query attention (GQA) and
the optional KV-cache and gradient-checkpointing forward paths (the latter two used by
Phases 6/7). These are the known-correct versions the owner diffs hand-written
variants against."""

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
        # getattr: this module also accepts a plain GPTConfig (which predates rope_base);
        # the default matches the value that was hard-coded before the knob existed.
        base = getattr(config, "rope_base", 10000.0)
        cos, sin = build_rope_cache(config.block_size, head_dim, base=base)
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


class NoPECausalSelfAttention(nn.Module):
    """Causal self-attention with NO positional signal at all (NoPE): raw q/k, no rotary
    tables, no learned wpe anywhere in the model. Position is inferable only from the
    causal mask (Kazemnejad et al., 2305.19466 — NoPE can match/beat explicit schemes and
    length-generalize; Kimi K3 ships it at frontier scale). Deliberately mirrors
    RoPECausalSelfAttention minus the rotation — same projection names/shapes, so the two
    A/B arms have identical param trees — and speaks the same KV-cache protocol (there is
    no position offset to track: cached decode just appends)."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor, kv_cache=None) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
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
        cos, sin = build_rope_cache(config.block_size, self.head_dim, base=config.rope_base)
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


def gdn_recurrent(q, k, v, alpha, beta):
    """Reference Gated DeltaNet recurrence — the O(T) sequential form. Obviously correct
    by construction; exists to be the test oracle for the chunkwise path, NOT to be
    trained with (a python loop over T is ~100x too slow).

    State S_t (per head) is a d_k x d_v outer-product memory:

        S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T
        o_t = S_t^T q_t

    i.e. the DeltaNet delta rule (erase the current read along k_t before writing v_t)
    with Mamba2-style scalar forget gate alpha_t. Shapes: q,k (B,H,T,Dk), v (B,H,T,Dv),
    alpha,beta (B,H,T). Returns (B,H,T,Dv)."""
    B, H, T, Dk = q.shape
    Dv = v.shape[-1]
    S = q.new_zeros(B, H, Dk, Dv)
    out = []
    eye = torch.eye(Dk, device=q.device, dtype=q.dtype)
    for t in range(T):
        k_t = k[:, :, t]                                   # (B,H,Dk)
        v_t = v[:, :, t]                                   # (B,H,Dv)
        a_t = alpha[:, :, t].unsqueeze(-1).unsqueeze(-1)    # (B,H,1,1)
        b_t = beta[:, :, t].unsqueeze(-1).unsqueeze(-1)     # (B,H,1,1)
        proj = eye - b_t * k_t.unsqueeze(-1) * k_t.unsqueeze(-2)   # (B,H,Dk,Dk)
        S = a_t * (proj @ S) + b_t * (k_t.unsqueeze(-1) * v_t.unsqueeze(-2))
        out.append((S.transpose(-2, -1) @ q[:, :, t].unsqueeze(-1)).squeeze(-1))
    return torch.stack(out, dim=2)


def gdn_chunkwise(q, k, v, alpha, beta, chunk: int = 64):
    """Chunk-parallel Gated DeltaNet — the trainable path. Mathematically identical to
    `gdn_recurrent` (enforced by tests/test_gdn.py to ~1e-4); T/chunk sequential steps
    instead of T.

    Derivation (all within one chunk, per head). Substituting S_t = A_t P_t with the
    cumulative decay A_t = prod_{i<=t} alpha_i turns the gated recurrence into an
    UNgated delta rule on P with rescaled write strength and query:

        P_t = (I - beta_t k_t k_t^T) P_{t-1} + (beta_t / A_t) k_t v_t^T
        o_t = P_t^T (A_t q_t)

    Writing the rank-1 update at step t as P_t = P_{t-1} + k_t d_t^T gives, for the
    matrix D of all d_t^T in the chunk,

        (I + diag(beta) M) D = diag(beta/A) V - diag(beta) K P_0,   M = strict_tril(K K^T)

    a unit-lower-triangular solve. Then P_C = P_0 + K^T D and, with L the INCLUSIVE
    lower-triangular mask (output uses the state after its own write),

        O = Qt P_0 + (L * (Qt K^T)) D,   Qt = A_t q_t.

    Numerics: beta/A_t grows as A_t decays, so the chunk must stay short and the solve
    must run in fp32 — hence chunk=64 and the .float() below. The decay gate is
    initialized near 1 (see GatedDeltaNet) which keeps A_t close to 1 across 64 steps."""
    B, H, T, Dk = q.shape
    Dv = v.shape[-1]
    if T % chunk != 0:
        raise ValueError(f"gdn_chunkwise: T={T} must be a multiple of chunk={chunk}")
    dtype = q.dtype
    # Promote to AT LEAST fp32 — the triangular solve and the beta/A_t rescale are the
    # numerically delicate steps and must not run in bf16/fp16. float64 inputs stay
    # float64 so tests can check the algebra at full precision rather than at the fp32
    # noise floor (~1e-7), which would hide a real error of that size.
    work = torch.float64 if dtype == torch.float64 else torch.float32
    q, k, v = q.to(work), k.to(work), v.to(work)
    alpha, beta = alpha.to(work), beta.to(work)
    nc = T // chunk

    q = q.view(B, H, nc, chunk, Dk)
    k = k.view(B, H, nc, chunk, Dk)
    v = v.view(B, H, nc, chunk, Dv)
    alpha = alpha.view(B, H, nc, chunk)
    beta = beta.view(B, H, nc, chunk)

    # Per-chunk cumulative decay A_t (A at the first position of a chunk is alpha_1
    # itself: the state entering the chunk is P_0, and step 1 already applies alpha_1).
    A = torch.cumprod(alpha.clamp_min(1e-6), dim=-1)              # (B,H,nc,chunk)

    eye = torch.eye(chunk, device=q.device, dtype=q.dtype)
    strict = torch.tril(torch.ones(chunk, chunk, device=q.device, dtype=q.dtype), -1)
    incl = torch.tril(torch.ones(chunk, chunk, device=q.device, dtype=q.dtype), 0)

    S = q.new_zeros(B, H, Dk, Dv)
    outs = []
    for c in range(nc):
        K, V, Q = k[:, :, c], v[:, :, c], q[:, :, c]              # (B,H,chunk,D*)
        b, a = beta[:, :, c], A[:, :, c]                          # (B,H,chunk)
        Qt = Q * a.unsqueeze(-1)                                  # A_t q_t
        b_over_a = (b / a).unsqueeze(-1)                           # beta_t / A_t
        bcol = b.unsqueeze(-1)

        M = (K @ K.transpose(-2, -1)) * strict                     # strict_tril(K K^T)
        lhs = eye + bcol * M                                       # unit lower triangular
        rhs = b_over_a * V - bcol * (K @ S)
        D = torch.linalg.solve_triangular(lhs, rhs, upper=False, unitriangular=True)

        attn = (Q @ K.transpose(-2, -1)) * a.unsqueeze(-1) * incl  # L * (Qt K^T)
        outs.append(Qt @ S + attn @ D)
        S = S + K.transpose(-2, -1) @ D                            # P_C
        S = S * a[..., -1].unsqueeze(-1).unsqueeze(-1)             # re-apply A_C: S = A_C P_C

    return torch.cat(outs, dim=2).view(B, H, T, Dv).to(dtype)


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet token mixer (Yang, Kautz & Hatamizadeh, ICLR 2025) — linear
    attention with the delta rule plus a Mamba2-style scalar forget gate. O(1) state per
    head instead of a growing KV cache, which is the whole reason to want it.

    Block contents follow the published one: short causal depthwise conv on q/k/v, SiLU
    then L2-normalisation on q/k (the delta rule needs unit keys to be stable), scalar
    per-head decay and write gates, and a SiLU output gate before the out-projection.

    DEVIATIONS from the paper, recorded so the verdict audit can weigh them:
    - The decay gate is alpha = sigmoid(W_a x + bias) with bias init +4.5 (alpha ~ 0.989)
      rather than the paper's exp(-softplus(dt) * exp(A)) discretisation. Monotone,
      bounded in (0,1), and keeps A_t near 1 across a 64-step chunk, which the chunkwise
      solve requires. Simpler to reason about; not identical.
    - No head-dim expansion (the paper widens v). Kept at head_dim so the A/B against
      attention is a token-mixer swap and not also a width change."""

    def __init__(self, config: VariantConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.chunk = config.gdn_chunk
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.conv = nn.Conv1d(
            3 * config.n_embd, 3 * config.n_embd, kernel_size=config.gdn_conv_kernel,
            groups=3 * config.n_embd, bias=False,
        )
        self.conv_pad = config.gdn_conv_kernel - 1
        self.a_proj = nn.Linear(config.n_embd, config.n_head, bias=True)
        self.b_proj = nn.Linear(config.n_embd, config.n_head, bias=True)
        self.g_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # Long-memory init: alpha ~ 0.989, beta ~ 0.5 (neutral write strength).
        nn.init.zeros_(self.a_proj.weight)
        nn.init.constant_(self.a_proj.bias, 4.5)
        nn.init.zeros_(self.b_proj.weight)
        nn.init.zeros_(self.b_proj.bias)

    def forward(self, x: torch.Tensor, kv_cache=None) -> torch.Tensor:
        if kv_cache is not None:
            raise NotImplementedError(
                "GatedDeltaNet has a recurrent state, not a KV cache; incremental "
                "decoding needs a separate state-carrying path (not built yet)"
            )
        B, T, C = x.shape
        qkv = self.qkv_proj(x).transpose(1, 2)                     # (B,3C,T)
        qkv = self.conv(F.pad(qkv, (self.conv_pad, 0)))            # causal depthwise
        qkv = qkv.transpose(1, 2)
        q, k, v = qkv.split(C, dim=2)
        shape = (B, T, self.n_head, self.head_dim)
        q = F.normalize(F.silu(q.view(shape)), dim=-1).transpose(1, 2)
        k = F.normalize(F.silu(k.view(shape)), dim=-1).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        alpha = torch.sigmoid(self.a_proj(x)).transpose(1, 2)      # (B,H,T)
        beta = torch.sigmoid(self.b_proj(x)).transpose(1, 2)       # (B,H,T)
        y = gdn_chunkwise(q, k, v, alpha, beta, chunk=self.chunk)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y * F.silu(self.g_proj(x)))


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
    pos: str = "learned"  # "learned" | "rope" | "nope" (no positional signal at all)
    mlp: str = "gelu"     # "gelu" | "swiglu"
    # Block layout. "pre": y = x + Module(Norm(x)) — the pre-norm block this file has
    # always built; the default is byte-identical to before this field existed. "peri":
    # y = x + Norm(Module(Norm(x))) — Peri-LN (arXiv 2502.02732): pre-norm PLUS an
    # output norm (own learnable scale, init ones, type follows `norm`) on each module
    # result before the residual add, bounding hidden-state variance growth; the exact
    # pre+post sandwich Gemma 2/3 ship. Deliberately implements ONLY the sublayer
    # wrapping: the paper's optional embedding-output norm is omitted (Gemma ships
    # without it, and it would add a second mechanism to the A/B), and the paper's
    # final-hidden-state norm already exists in every variant here as ln_f.
    block_norm: str = "pre"
    # None -> classic multi-head attention (fused c_attn), bit-identical to before this
    # field existed. Set to a divisor of n_head for grouped-query attention (1 == MQA).
    n_kv_head: int | None = None
    # RoPE frequency base (theta). 10000.0 is the value that was hard-coded before this
    # field existed; larger bases are the lever for context extension (PI/YaRN stage).
    rope_base: float = 10000.0
    # None -> every layer is global attention (the dense baseline; bit-identical to
    # before this field existed). Set to N for the Kimi-Linear-style layerwise hybrid:
    # every Nth layer (1-indexed, so the LAST of each group) stays global attention and
    # the other N-1 are GatedDeltaNet. N=4 is the published 3:1 linear:full ratio.
    # NOTE `pos` then applies only to the surviving global layers — GDN layers carry no
    # positional encoding at all, position enters through the recurrence. So
    # pos="nope" + hybrid_every=4 IS the Kimi Linear configuration, and is the arm that
    # retests the NoPE conditional left open by docs/nope-verdict-audit.md.
    hybrid_every: int | None = None
    gdn_chunk: int = 64        # chunk-parallel block length; see gdn_chunkwise numerics
    gdn_conv_kernel: int = 4   # short causal depthwise conv on q/k/v, as published


def _make_norm(kind: str, dim: int) -> nn.Module:
    return RMSNorm(dim) if kind == "rms" else nn.LayerNorm(dim)


class VariantBlock(nn.Module):
    def __init__(self, config: VariantConfig, layer_idx: int = 0) -> None:
        super().__init__()
        # getattr: this module is sometimes handed configs unpickled from checkpoints
        # written before block_norm existed; the default reproduces that era exactly.
        block_norm = getattr(config, "block_norm", "pre")
        if block_norm not in ("pre", "peri"):
            raise ValueError(
                f"unknown block_norm {block_norm!r}: expected 'pre' or 'peri'"
            )
        self.block_norm = block_norm
        self.ln_1 = _make_norm(config.norm, config.n_embd)
        # Hybrid routing: with hybrid_every=N, layers 0..N-2 of each group of N are
        # GatedDeltaNet and layer N-1 (the last) stays global attention.
        hybrid_every = getattr(config, "hybrid_every", None)
        self.is_linear = (
            hybrid_every is not None and (layer_idx + 1) % hybrid_every != 0
        )
        if self.is_linear:
            self.attn = GatedDeltaNet(config)
        elif getattr(config, "n_kv_head", None) is not None:
            self.attn = GQAAttention(config)  # rope-only; asserts on other pos values
        elif config.pos == "rope":
            self.attn = RoPECausalSelfAttention(config)
        elif config.pos == "nope":
            self.attn = NoPECausalSelfAttention(config)
        elif config.pos == "learned":
            self.attn = CausalSelfAttention(config)
        else:
            raise ValueError(
                f"unknown pos {config.pos!r}: expected 'learned', 'rope', or 'nope'"
            )
        self.ln_2 = _make_norm(config.norm, config.n_embd)
        self.mlp = SwiGLUMLP(config) if config.mlp == "swiglu" else MLP(config)
        if block_norm == "peri":
            # Peri-LN output norms (one per sublayer, own learnable scales — RMSNorm /
            # LayerNorm init their gains to ones, the standard init). No RNG is drawn,
            # so shared-param init is bit-identical across "pre"/"peri" at equal seed.
            self.ln_1_post = _make_norm(config.norm, config.n_embd)
            self.ln_2_post = _make_norm(config.norm, config.n_embd)

    def forward(self, x: torch.Tensor, kv_cache=None) -> torch.Tensor:
        if kv_cache is not None:
            attn_out = self.attn(self.ln_1(x), kv_cache=kv_cache)
        else:
            attn_out = self.attn(self.ln_1(x))
        if self.block_norm == "peri":
            attn_out = self.ln_1_post(attn_out)
        x = x + attn_out
        mlp_out = self.mlp(self.ln_2(x))
        if self.block_norm == "peri":
            mlp_out = self.ln_2_post(mlp_out)
        x = x + mlp_out
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
            h=nn.ModuleList(
                [VariantBlock(config, layer_idx=i) for i in range(config.n_layer)]
            ),
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
            # rope/nope attentions speak the cache protocol; the learned-pos block does not
            assert self.config.pos in ("rope", "nope"), \
                "KV cache requires the RoPE or NoPE block"
        x = self.transformer.wte(idx)
        if self.config.pos == "learned":
            pos = torch.arange(T, device=idx.device)
            x = x + self.transformer.wpe(pos)
        x = self.transformer.drop(x)
        for i, block in enumerate(self.transformer.h):
            if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                # Training path only: kv_cache is always None here (caching asserts
                # pos=="rope" and runs under no_grad in eval/generation).
                assert kv_cache is None, "kv_cache is unsupported under gradient checkpointing"
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
