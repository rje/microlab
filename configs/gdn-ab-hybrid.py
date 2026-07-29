"""GDN/KDA hybrid A/B — "the big one" on the ablation ladder (docs/arch-review-2026.md).

Both arms are the adopted recipe (Peri-LN + Muon + RoPE + SwiGLU) at ~124M / 4500 steps.
The ONLY difference is `hybrid_every`: the dense arm is all global attention; the hybrid
arm replaces 3 of every 4 layers with GatedDeltaNet (the published Kimi-Linear 3:1 ratio).

TWO CONFOUNDS, recorded up front so the verdict audit weighs them:

1. PARAMS ARE NOT MATCHED. Hybrid 115.11M vs dense 109.59M (+5.0%), entirely from
   GatedDeltaNet's SiLU output gate (one extra n_embd^2 per linear layer). We run the
   PUBLISHED block rather than crippling it for parity, so: a hybrid LOSS is decisive,
   while a hybrid WIN smaller than what +5% params would buy is INCONCLUSIVE and needs a
   param-matched rerun.

2. THROUGHPUT IS NOT MEASURABLE HERE. Benchmarked 262 ms/step vs dense 91 ms/step at
   batch 4 x 1024 — the hybrid is 2.9x SLOWER. That is our pure-PyTorch chunkwise scan
   (T/64 sequential triangular solves) against a fused SDPA kernel, NOT a property of
   linear attention, whose whole selling point is the opposite. GDN's efficiency claim
   needs Triton kernels we have not written. This lane answers "does the 3:1 hybrid cost
   quality at 124M?" and CANNOT answer "is it faster or cheaper."

Layer correctness is gated by tests/test_gdn.py: the chunkwise path is checked against a
sequential reference at float64 to ~1e-8, plus a causality test.

Per docs/periln-verdict.md the cross-seed noise band is ~0.013 nats at this scale, so a
single seed pair CANNOT settle this lane; seed-1338 copies must run before any verdict.
"""

from microlab.train.config import RunConfig

config = RunConfig(
    # model (~124M with vocab 32k; Peri-LN block layout — see formulation note above)
    vocab_size=32000,
    block_size=1024,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.0,
    norm="rms",
    pos="rope",
    mlp="swiglu",
    block_norm="peri",
    hybrid_every=4,
    # optim / schedule — lab-standard Muon (matrices) + AdamW (embeddings/norms; the
    # extra peri norm scales are 1-D so they land on AdamW automatically),
    # values straight from the muon-ab arms
    optimizer="muon",
    muon_lr=0.02,
    lr=6e-4,
    min_lr=6e-5,
    weight_decay=0.1,
    grad_clip=1.0,
    warmup_steps=150,
    max_steps=4500,
    lr_decay_steps=4500,
    # data / io
    batch_size=10,
    grad_accum=16,
    compile=True,
    # "-no-cudagraphs" is REQUIRED: plain max-autotune captures CUDA graphs, whose static
    # memory pool is incompatible with the tied lm_head/wte weight under grad accumulation
    # — it crashes on the first backward (see configs/1b.py).
    compile_mode="max-autotune-no-cudagraphs",
    eval_interval=250,   # 18 matched-step val-loss points per arm (same seeded batches)
    eval_iters=50,       # 50 x 40 x 1024 ~= 2M val tokens per point
    ckpt_interval=500,
    ckpt_keep=2,         # ablation run: rolling recovery only, no permanent trajectory
    log_interval=25,
    out_dir="runs/gdn-ab-hybrid",
    device="cuda",
    dtype="bfloat16",
    # seed is the SINGLE source of truth for init AND data order (both arms share it).
    # Variance measurement (finding #9's std claim needs 2-3 seeds/arm): the orchestrator
    # copies this file changing ONLY seed and out_dir (e.g. seed=1338,
    # out_dir="runs/periln-ab-peri-s1338"), then passes every run dir to
    # scripts/analyze_periln_ab.py side by side.
    seed=1337,
)
