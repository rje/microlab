"""Code-specialist data-lane BASE config — 124M on the-stack code corpus.

The architecture is FROZEN to the adopted recipe so these lanes vary DATA only:
Peri-LN + Muon + RoPE + SwiGLU + the 3:1 GDN hybrid (verdict 3, adopted 2026-07-30).

Differences from the FineWeb ablation configs, all forced by the corpus:
  - vocab_size 49152 (code-49k tokenizer, ~44% fewer tokens/byte on code) not 32000
  - data-dir data/shards/code-stack-30b (27.35B train / 28.7M val tokens)

Duration is 15000 steps = 2.46B tokens = 0.99x Chinchilla for 124M, per the protocol change
of 2026-07-30: a 4500-step lane is 0.30x Chinchilla and can invert its own verdict.

DEPENDENCY: Peri-LN is currently under retest at 15000 steps (runs/periln-long-*). If that
retest overturns verdict 2, every lane built on this base inherits the change — but since
all data arms share the architecture, the RELATIVE data comparison stays valid.

Shard->language map (verified by decoding samples, 100M tokens per shard):
  train-00000..00099 Python | 00100..00199 JavaScript | 00200..00273 TypeScript
so language-mix arms are made by subsetting the manifest, not by rebuilding the corpus.
"""

from microlab.train.config import RunConfig

config = RunConfig(
    # model (~124M with vocab 32k; Peri-LN block layout — see formulation note above)
    vocab_size=49152,
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
    out_dir="runs/code-base",
    device="cuda",
    dtype="bfloat16",
    # seed is the SINGLE source of truth for init AND data order (both arms share it).
    # Variance measurement (finding #9's std claim needs 2-3 seeds/arm): the orchestrator
    # copies this file changing ONLY seed and out_dir (e.g. seed=1338,
    # out_dir="runs/periln-ab-peri-s1338"), then passes every run dir to
    # scripts/analyze_periln_ab.py side by side.
    seed=1337,
)
