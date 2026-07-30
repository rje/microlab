"""Peri-LN-vs-Pre-LN A/B — Pre-LN arm (control; ~124M, GPT-2-small class: 12L x 768,
RoPE + RMSNorm + SwiGLU, Muon). IDENTICAL to configs/periln-ab-peri.py except
`block_norm` and out_dir; sizing is the muon-ab recipe reused wholesale (batch 40 x
accum 4 x block 1024, 4500 steps ~737M tokens) so curves are comparable across A/Bs.

WHY THIS ABLATION RUNS FIRST: docs/arch-review-2026.md finding #9 — Peri-LN (wrap each
sublayer: y = x + Norm(Module(Norm(x)))) beats Pre-LN at 400M-3.2B on loss AND
downstream, and cuts seed-to-seed benchmark std by MORE THAN HALF (1.5B: +-1.22 ->
+-0.21). The variance reduction is the point: if it replicates here, every later
ablation lane runs on the lower-noise layout and needs fewer seeds. This arm is the
control: the standard pre-norm block, y = x + Module(Norm(x)).

Formulation note: the peri arm implements the SUBLAYER-WRAPPING core of Peri-LN (arXiv
2502.02732) — exactly Gemma 2/3's pre+post sandwich. The paper's optional
embedding-output norm is omitted (single-mechanism A/B; Gemma ships without it) and the
final-state norm already exists as ln_f in both arms. Unlike the nope A/B, the arms'
param trees are NOT identical: peri adds 2 norm scales per layer (12 x 2 x 768 = 18,432
params, +0.015%) initialized to ones — inherent to the ablation, negligible capacity.

After both arms finish: scripts/analyze_periln_ab.py (accepts all run dirs, including
multi-seed variants).

    python scripts/pretrain.py configs/periln-ab-pre.py --data-dir data/shards/fineweb-100bt
"""

from microlab.train.config import RunConfig

config = RunConfig(
    # model (~124M with vocab 32k; standard pre-norm block layout)
    vocab_size=32000,
    block_size=1024,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.0,
    norm="rms",
    pos="rope",
    mlp="swiglu",
    block_norm="pre",
    # optim / schedule — lab-standard Muon (matrices) + AdamW (embeddings/norms — the
    # peri arm's extra norm scales are 1-D, so they land on AdamW there too),
    # values straight from the muon-ab arms
    optimizer="muon",
    muon_lr=0.02,
    lr=6e-4,
    min_lr=6e-5,
    weight_decay=0.1,
    grad_clip=1.0,
    warmup_steps=150,
    max_steps=15000,
    lr_decay_steps=15000,
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
    ckpt_interval=1000,
    ckpt_keep=2,         # ablation run: rolling recovery only, no permanent trajectory
    log_interval=25,
    out_dir="runs/periln-long-pre",
    device="cuda",
    dtype="bfloat16",
    # seed is the SINGLE source of truth for init AND data order (both arms share it).
    # Variance measurement (finding #9's std claim needs 2-3 seeds/arm): the orchestrator
    # copies this file changing ONLY seed and out_dir (e.g. seed=1338,
    # out_dir="runs/periln-ab-pre-s1338"), then passes every run dir to
    # scripts/analyze_periln_ab.py side by side.
    seed=1337,
)
