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


def kda_recurrent(q, k, v, alpha, beta):
    """Reference KDA (Kimi Delta Attention) recurrence — the oracle for the fused kernel.

    KDA generalises Gated DeltaNet by making the forget gate PER-CHANNEL instead of one
    scalar per head. That is the whole difference, and it is the difference that matters
    for NoPE-on-globals: if the recurrence is what carries positional information, a
    diagonal gate of width K carries ~K times more of it than a single scalar.

        S_t = Diag(alpha_t) (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T
        o_t = S_t^T q_t

    Shapes: q,k (B,H,T,Dk), v (B,H,T,Dv), alpha (B,H,T,Dk) <- per-channel, beta (B,H,T).
    Set alpha to a broadcast scalar per head and this reduces exactly to gdn_recurrent,
    which tests/test_kda.py asserts."""
    B, H, T, Dk = q.shape
    Dv = v.shape[-1]
    S = q.new_zeros(B, H, Dk, Dv)
    eye = torch.eye(Dk, device=q.device, dtype=q.dtype)
    out = []
    for t in range(T):
        k_t, v_t = k[:, :, t], v[:, :, t]
        a_t = alpha[:, :, t].unsqueeze(-1)                      # (B,H,Dk,1) diagonal
        b_t = beta[:, :, t].unsqueeze(-1).unsqueeze(-1)
        proj = eye - b_t * k_t.unsqueeze(-1) * k_t.unsqueeze(-2)
        S = a_t * (proj @ S) + b_t * (k_t.unsqueeze(-1) * v_t.unsqueeze(-2))
        out.append((S.transpose(-2, -1) @ q[:, :, t].unsqueeze(-1)).squeeze(-1))
    return torch.stack(out, dim=2)


def _fla_kda(q, k, v, alpha, beta):
    """Fused KDA kernel. alpha is per-channel (B,H,T,Dk), linear space; fla wants log."""
    if q.dtype == torch.float64 or not q.is_cuda:
        return None
    try:
        from fla.ops.kda import chunk_kda
    except ImportError:
        return None
    dt = q.dtype if q.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
    o = chunk_kda(
        q=q.permute(0, 2, 1, 3).contiguous().to(dt),
        k=k.permute(0, 2, 1, 3).contiguous().to(dt),
        v=v.permute(0, 2, 1, 3).contiguous().to(dt),
        g=torch.log(alpha.clamp_min(1e-30)).permute(0, 2, 1, 3).contiguous().float(),
        beta=beta.permute(0, 2, 1).contiguous().to(dt),
        scale=1.0,
    )
    o = o[0] if isinstance(o, tuple) else o
    return o.permute(0, 2, 1, 3).to(q.dtype)


def _fla_gdn(q, k, v, alpha, beta):
    """Fused Triton path (flash-linear-attention). Returns None when unavailable or when
    the caller wants exact float64, so the reference stays the oracle.

    Equivalence with gdn_recurrent is verified in tests/test_gdn_fused.py: the kernel lands
    at 0.74x the bf16 REPRESENTATION floor against our float64 reference, with correlation
    1.000000 over 1024 steps — same recurrence, computed in bf16.

    Why this exists: our gdn_chunkwise is a python loop of T/chunk sequential iterations
    that must run in fp32, because torch.linalg.solve_triangular has no bfloat16 CUDA
    kernel. Measured 23-31x slower than fused, and the gap does NOT shrink with context.
    A correctness oracle cannot be a training kernel."""
    # Decline on CPU (Triton is CUDA-only) and on float64 (the oracle path). Returning
    # None rather than raising keeps the reference implementation as a real fallback, so
    # CPU tests and debugging still exercise the same module.
    if q.dtype == torch.float64 or not q.is_cuda:
        return None
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    except ImportError:
        return None
    dt = q.dtype if q.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
    o = chunk_gated_delta_rule(
        q=q.permute(0, 2, 1, 3).contiguous().to(dt),
        k=k.permute(0, 2, 1, 3).contiguous().to(dt),
        v=v.permute(0, 2, 1, 3).contiguous().to(dt),
        g=torch.log(alpha.clamp_min(1e-30)).permute(0, 2, 1).contiguous().float(),
        beta=beta.permute(0, 2, 1).contiguous().to(dt),
        scale=1.0,   # q/k are already L2-normalised; fla would otherwise apply 1/sqrt(K)
    )
    o = o[0] if isinstance(o, tuple) else o
    return o.permute(0, 2, 1, 3).to(q.dtype)


def gdn_step(q, k, v, alpha, beta, S):
    """Single-token Gated DeltaNet step for incremental decoding. One application of the
    recurrence, O(1) in context length — this is the whole point of the architecture.
    q,k (B,H,1,Dk), v (B,H,1,Dv), alpha,beta (B,H,1), S (B,H,Dk,Dv). Returns (out, S_new).

    Deliberately NOT gdn_chunkwise with T=1: that would pad to a full 64-wide chunk and
    build a 64x64 solve to advance one position."""
    work = torch.float64 if q.dtype == torch.float64 else torch.float32
    dtype = q.dtype
    k_t = k[:, :, 0].to(work)                                # (B,H,Dk)
    v_t = v[:, :, 0].to(work)
    q_t = q[:, :, 0].to(work)
    a = alpha[:, :, 0].to(work).unsqueeze(-1).unsqueeze(-1)  # (B,H,1,1)
    b = beta[:, :, 0].to(work).unsqueeze(-1).unsqueeze(-1)
    Sw = S.to(work)
    # S = alpha (I - beta k k^T) S + beta k v^T, computed without forming (Dk,Dk):
    #   (I - beta k k^T) S = S - beta k (k^T S)
    kS = torch.einsum("bhd,bhdv->bhv", k_t, Sw).unsqueeze(-2)          # (B,H,1,Dv)
    Sn = a * (Sw - b * k_t.unsqueeze(-1) * kS) + b * (k_t.unsqueeze(-1) * v_t.unsqueeze(-2))
    out = torch.einsum("bhd,bhdv->bhv", q_t, Sn).unsqueeze(2)          # (B,H,1,Dv)
    return out.to(dtype), Sn.to(S.dtype)


def gdn_chunkwise(q, k, v, alpha, beta, chunk: int = 64, S0=None, return_state: bool = False):
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

    Numerics — the NON-obvious part, and the reason this is written with decay RATIOS
    rather than the textbook beta/A_t. Dividing by the cumulative decay A_t is
    catastrophic: with a learned gate settling near alpha~0.5, A_64 reaches 1e-23 and
    beta/A_t reaches 1e22. The forward pass survives (the A_t in the output term cancels
    the 1/A_t in the write term) but the backward does not, and training NaNs. Measured,
    not hypothesised — see the 2026-07-29 smoke failure.

    So every decay here appears as a ratio A_t/A_j with j <= t, which is bounded by 1 for
    alpha <= 1, computed in log space. Nothing in the solve or the output can exceed the
    scale of the inputs, for any gate value:

        (I + diag(beta) (G_strict * K K^T)) E = diag(beta) V - diag(beta*A) K S_0
        O     = diag(A) Q S_0 + ((Q K^T) * G_incl) E
        S_new = A_C S_0 + K^T diag(A_C/A) E          where G[t,j] = A_t/A_j

    fp32 is still the floor for the triangular solve (see the .to(work) below)."""
    B, H, T, Dk = q.shape
    Dv = v.shape[-1]
    T_real = T
    if T % chunk != 0:
        # Right-pad to a whole number of chunks. The padding is an exact no-op on the
        # state, not an approximation: with k=0 the delta projection (I - beta k k^T) is
        # the identity and the write beta k v^T is zero, and with alpha=1 there is no
        # decay — so the state after the last real token is carried through untouched.
        # Padded outputs are sliced off below. Causality (tested) guarantees real
        # positions cannot see the padding. Needed because generation feeds T=1..n
        # token by token; block_size is always a multiple of chunk during training.
        pad = chunk - (T % chunk)
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        alpha = F.pad(alpha, (0, pad), value=1.0)
        beta = F.pad(beta, (0, pad), value=0.0)
        T = T + pad
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

    # Cumulative decay per chunk, in log space so ratios are exact subtractions.
    logA = torch.cumsum(torch.log(alpha.clamp_min(1e-30)), dim=-1)   # (B,H,nc,chunk)
    A = torch.exp(logA)

    eye = torch.eye(chunk, device=q.device, dtype=q.dtype)
    strict = torch.tril(torch.ones(chunk, chunk, device=q.device, dtype=q.dtype), -1)
    incl = torch.tril(torch.ones(chunk, chunk, device=q.device, dtype=q.dtype), 0)

    S = q.new_zeros(B, H, Dk, Dv) if S0 is None else S0.to(q.dtype).clone()
    outs = []
    for c in range(nc):
        K, V, Q = k[:, :, c], v[:, :, c], q[:, :, c]              # (B,H,chunk,D*)
        b, a, la = beta[:, :, c], A[:, :, c], logA[:, :, c]       # (B,H,chunk)
        bcol = b.unsqueeze(-1)

        # G[t,j] = A_t / A_j, bounded by 1 for j <= t. Never form 1/A_t.
        # clamp_max(0) is a NO-OP on the entries we keep (logA is non-increasing, so
        # logA_t - logA_j <= 0 whenever j <= t) and is essential for the ones we discard:
        # above the diagonal the difference is large and POSITIVE, exp() gives +inf, and
        # inf * 0 from the triangular mask is NaN. Masking after exp is not enough.
        G = torch.exp((la.unsqueeze(-1) - la.unsqueeze(-2)).clamp_max(0.0))
        M = (K @ K.transpose(-2, -1)) * (G * strict)
        lhs = eye + bcol * M                                      # unit lower triangular
        rhs = bcol * V - (b * a).unsqueeze(-1) * (K @ S)
        E = torch.linalg.solve_triangular(lhs, rhs, upper=False, unitriangular=True)

        attn = (Q @ K.transpose(-2, -1)) * (G * incl)
        outs.append(a.unsqueeze(-1) * (Q @ S) + attn @ E)
        # S_new = A_C S_0 + K^T diag(A_C / A) E
        ratio = torch.exp(la[..., -1:] - la)                      # (B,H,chunk), all <= 1
        S = a[..., -1].unsqueeze(-1).unsqueeze(-1) * S \
            + K.transpose(-2, -1) @ (ratio.unsqueeze(-1) * E)

    y = torch.cat(outs, dim=2).view(B, H, T, Dv)[:, :, :T_real].to(dtype)
    return (y, S) if return_state else y


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
        # Fused kernel for training; the pure-PyTorch scan remains the oracle and the
        # fallback. Set gdn_fused=False to force the reference path (tests, debugging).
        self.fused = getattr(config, "gdn_fused", True)
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.conv = nn.Conv1d(
            3 * config.n_embd, 3 * config.n_embd, kernel_size=config.gdn_conv_kernel,
            groups=3 * config.n_embd, bias=False,
        )
        self.conv_pad = config.gdn_conv_kernel - 1
        self.gate = getattr(config, "gdn_gate", "scalar")
        if self.gate not in ("scalar", "channel"):
            raise ValueError(f"unknown gdn_gate {self.gate!r}: expected 'scalar' or 'channel'")
        # KDA emits one decay per (head, head_dim) channel; GDN one per head.
        a_out = config.n_head * self.head_dim if self.gate == "channel" else config.n_head
        self.a_proj = nn.Linear(config.n_embd, a_out, bias=True)
        self.b_proj = nn.Linear(config.n_embd, config.n_head, bias=True)
        self.g_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.reset_gate_parameters()

    def reset_gate_parameters(self) -> None:
        """Long-memory init: alpha ~ 0.989 (bias 4.5), beta ~ 0.5 (neutral write). Called
        from __init__ AND re-called by VariantGPT after its generic apply(_init_weights),
        which otherwise overwrites these with normal(0,0.02)/zeros and starts alpha at
        ~0.5 — a decay so fast the chunk-cumulative product underflows."""
        nn.init.zeros_(self.a_proj.weight)
        nn.init.constant_(self.a_proj.bias, 4.5)
        nn.init.zeros_(self.b_proj.weight)
        nn.init.zeros_(self.b_proj.bias)

    def forward(self, x: torch.Tensor, kv_cache=None) -> torch.Tensor:
        B, T, C = x.shape
        cache, layer = (kv_cache if kv_cache is not None else (None, None))
        qkv = self.qkv_proj(x).transpose(1, 2)                     # (B,3C,T)
        if cache is None:
            qkv = self.conv(F.pad(qkv, (self.conv_pad, 0)))        # causal depthwise
        else:
            # Incremental: instead of zero-padding, prepend the last conv_pad raw inputs
            # so the conv sees real history. Zero-padding mid-stream would silently
            # change the output of every decoded token.
            hist = cache.conv_hist(layer, B, 3 * C, self.conv_pad, qkv.dtype, qkv.device)
            stream = torch.cat([hist, qkv], dim=2)
            qkv = self.conv(stream)
            cache.set_conv_hist(layer, stream[:, :, -self.conv_pad:])
        qkv = qkv.transpose(1, 2)
        q, k, v = qkv.split(C, dim=2)
        shape = (B, T, self.n_head, self.head_dim)
        q = F.normalize(F.silu(q.view(shape)), dim=-1).transpose(1, 2)
        k = F.normalize(F.silu(k.view(shape)), dim=-1).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        if self.gate == "channel":                                  # KDA: (B,H,T,Dh)
            alpha = torch.sigmoid(self.a_proj(x)).view(B, T, self.n_head, self.head_dim)
            alpha = alpha.transpose(1, 2)
        else:                                                       # GDN: (B,H,T)
            alpha = torch.sigmoid(self.a_proj(x)).transpose(1, 2)
        beta = torch.sigmoid(self.b_proj(x)).transpose(1, 2)       # (B,H,T)
        if cache is None:
            if self.gate == "channel":
                y = _fla_kda(q, k, v, alpha, beta) if self.fused else None
                if y is None:
                    y = kda_recurrent(q, k, v, alpha, beta)   # oracle fallback (slow)
            else:
                y = _fla_gdn(q, k, v, alpha, beta) if self.fused else None
                if y is None:
                    y = gdn_chunkwise(q, k, v, alpha, beta, chunk=self.chunk)
        else:
            if self.gate == "channel":
                raise NotImplementedError(
                    "KDA incremental decode needs a per-channel gdn_step; use gdn_gate="
                    "'scalar' for cached generation until that exists")
            S = cache.gdn_state(layer, B, self.n_head, self.head_dim, q.dtype, q.device)
            if T == 1:
                y, S = gdn_step(q, k, v, alpha, beta, S)
            else:                                                   # prefill
                y, S = gdn_chunkwise(q, k, v, alpha, beta, chunk=self.chunk,
                                     S0=S, return_state=True)
            cache.set_gdn_state(layer, S)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y * F.silu(self.g_proj(x)))


class MLAAttention(nn.Module):
    """Multi-head Latent Attention (DeepSeek-V2), NoPE variant — the global-attention layer
    of the Kimi Linear pattern.

    Instead of SHARING key/value heads across query groups (GQA), MLA COMPRESSES K/V into a
    single low-rank latent c_kv of width `mla_kv_lora`, caches only that, and up-projects it
    back into per-head K and V. Every head gets its own K/V; the bottleneck is the latent
    rank, not the head count.

    Because these layers are NoPE (position is carried by the KDA recurrence), DeepSeek's
    decoupled-RoPE split is unnecessary and omitted: there is no position-dependent term that
    would block folding the up-projection, so the cache is exactly `mla_kv_lora` values per
    token. At our shape that is 512 — IDENTICAL to GQA(2) — while GQA(2) shares one K/V head
    across 7 queries. Costs ~25% more params per layer (9.18M vs 7.34M).

    Cache note: `kv_cache` here stores the LATENT, not K/V, so it does not speak the KVCache
    protocol. Incremental decode raises rather than silently mis-caching."""

    def __init__(self, config: VariantConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.kv_lora = config.mla_kv_lora
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.kv_a_proj = nn.Linear(config.n_embd, self.kv_lora, bias=False)
        self.kv_a_norm = RMSNorm(self.kv_lora)
        self.kv_b_proj = nn.Linear(self.kv_lora, config.n_head * 2 * self.head_dim,
                                   bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor, kv_cache=None) -> torch.Tensor:
        if kv_cache is not None:
            raise NotImplementedError(
                "MLA caches the compressed latent, not K/V; it does not speak the KVCache "
                "protocol. A latent-caching decode path is not built yet.")
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        c_kv = self.kv_a_norm(self.kv_a_proj(x))                  # (B,T,kv_lora) <- cached
        kv = self.kv_b_proj(c_kv).view(B, T, self.n_head, 2 * self.head_dim)
        k, v = kv.transpose(1, 2).split(self.head_dim, dim=-1)    # (B,H,T,Dh) each, DISTINCT
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.o_proj(y.transpose(1, 2).contiguous().view(B, T, C))


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
    gdn_fused: bool = True     # use the fused Triton kernel; False forces the reference
    # "scalar" -> Gated DeltaNet (one forget gate per head). "channel" -> KDA (Kimi Delta
    # Attention): a per-channel diagonal gate, ~head_dim times the capacity. Kimi Linear's
    # results come from the channel gate; our first hybrid used scalar and we then drew
    # conclusions about the KDA lineage from it.
    gdn_gate: str = "scalar"
    # Global-attention layer type in a hybrid: "gqa" (n_kv_head sharing) or "mla" (latent
    # compression, DeepSeek-V2 / Kimi Linear). With NoPE globals MLA needs no decoupled
    # RoPE, so its cache is exactly mla_kv_lora values/token.
    global_attn: str = "gqa"
    mla_kv_lora: int = 512


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
        elif getattr(config, "global_attn", "gqa") == "mla":
            self.attn = MLAAttention(config)
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
        # Fused linear+cross-entropy (Liger). Off by default so existing behaviour and
        # every eval that reads logits is unchanged; the trainer turns it on.
        self.fused_ce = False
        self.training_logits = False   # force the logits path even when fused_ce is set
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
        # MUST come after apply(): _init_weights treats every nn.Linear the same and
        # would clobber the gate init below (it did — the first GDN training attempt
        # NaN'd because alpha landed at sigmoid(0)~0.5 instead of ~0.99, driving the
        # cumulative chunk decay to 1e-23). Any module needing an init that survives the
        # generic pass has to re-apply it here.
        for m in self.modules():
            if isinstance(m, GatedDeltaNet):
                m.reset_gate_parameters()

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
        loss = None
        if targets is not None and self.fused_ce and x.is_cuda and not self.training_logits:
            # Fused linear+CE never materialises [B,T,V]. That tensor chain was ~half of
            # all training memory: measured 497 KB/token at V=49152, versus ~86 KB/token
            # for ALL 24 layers of checkpointed activations combined. Liger cuts the loss
            # path 73% (4.42 -> 1.21 GB at 8192 tokens) at bf16-equivalent accuracy —
            # validated against fp32 truth, not against our own bf16 path: median relative
            # gradient error 2.37e-3 vs our naive path's 1.74e-3, both corr > 0.99999,
            # identical loss to six decimals. Returns logits=None; callers that need
            # logits (generation, evals) simply pass targets=None.
            from liger_kernel.transformers.fused_linear_cross_entropy import (
                LigerFusedLinearCrossEntropyLoss,
            )
            loss = LigerFusedLinearCrossEntropyLoss()(
                self.lm_head.weight, x.reshape(-1, x.size(-1)), targets.reshape(-1))
            return None, loss
        logits = self.lm_head(x)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
