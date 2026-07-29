"""Peri-LN-vs-Pre-LN A/B — Peri-LN arm (treatment; ~124M, GPT-2-small class: 12L x 768,
RoPE + RMSNorm + SwiGLU, Muon). IDENTICAL to configs/periln-ab-pre.py except
`block_norm` and out_dir: every sublayer becomes y = x + Norm(Module(Norm(x))) — the
standard pre-norm PLUS an RMSNorm (own learnable scale, init ones) on the module output
before the residual add (Peri-LN, arXiv 2502.02732; Gemma 2/3's pre+post sandwich).

WHY THIS ABLATION RUNS FIRST: docs/arch-review-2026.md finding #9 — in controlled
5-seed runs Peri-LN beats Pre-LN at 400M-3.2B on loss AND downstream (1.5B: loss 3.18
vs 3.29, bench avg 56.55 vs 53.71) and cuts seed-to-seed benchmark std by MORE THAN
HALF, by bounding hidden-state variance growth across depth. If the variance reduction
replicates at 124M, every later ablation lane runs on this layout and needs fewer
seeds — this lane calibrates all of them.

Formulation note: this implements the SUBLAYER-WRAPPING core of Peri-LN only. The
paper's optional embedding-output norm is omitted (single-mechanism A/B; the shipped
Gemma 2/3 analog omits it too) and the paper's final-state norm already exists as ln_f
in both arms. Param trees are NOT identical across arms: this arm has 2 extra norm
scales per layer (12 x 2 x 768 = 18,432 params, +0.015%, all init ones, on AdamW in the
Muon hybrid) — inherent to the ablation, negligible capacity.

After both arms finish: scripts/analyze_periln_ab.py (accepts all run dirs, including
multi-seed variants).

    python scripts/pretrain.py configs/periln-ab-peri.py --data-dir data/shards/fineweb-100bt
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
    batch_size=40,
    grad_accum=4,
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
    out_dir="runs/periln-ab-peri-s1339",
    device="cuda",
    dtype="bfloat16",
    # seed is the SINGLE source of truth for init AND data order (both arms share it).
    # Variance measurement (finding #9's std claim needs 2-3 seeds/arm): the orchestrator
    # copies this file changing ONLY seed and out_dir (e.g. seed=1338,
    # out_dir="runs/periln-ab-peri-s1338"), then passes every run dir to
    # scripts/analyze_periln_ab.py side by side.
    seed=1339,
)
