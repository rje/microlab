"""Muon A/B validation — Muon arm (~124M params, GPT-2-small class: 12L x 768, RoPE +
RMSNorm + SwiGLU). IDENTICAL to configs/muon-ab-adamw.py except `optimizer` and out_dir;
compare val-loss curves at matched steps (evals use the same seeded batches in both arms).

Muon runs on the transformer-block matrices; embeddings (incl. the tied wte/lm_head
tensor) and norm gains stay on AdamW at the lr below. muon_lr=0.02 is the reference
impl's default for our update convention (orthogonalized momentum scaled by
max(1, rows/cols)**0.5): Muon's near-orthogonal updates have unit-scale singular values,
so its LR sits ~30x above AdamW's — 0.02 is what both the Muon repo and the nanoGPT
speedrun settled on at this model scale. Both param groups follow the same
warmup+cosine schedule shape, each at its own scale.

    python scripts/pretrain.py configs/muon-ab-muon.py --data-dir data/shards/fineweb-100bt
"""

from microlab.train.config import RunConfig

config = RunConfig(
    # model (~124M with vocab 32k; RoPE means no positional-embedding params)
    vocab_size=32000,
    block_size=1024,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.0,
    norm="rms",
    pos="rope",
    mlp="swiglu",
    # optim / schedule — AdamW settings identical to the other arm (they drive the
    # embeddings/norms aux group here); muon_lr drives the matrix group.
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
    eval_interval=250,   # 18 val-loss points per arm (~5.5min each at 1.3 s/step)
    eval_iters=50,       # 50 x 40 x 1024 ~= 2M val tokens; same seeded batches both arms
    ckpt_interval=500,
    ckpt_keep=2,         # validation run: rolling recovery only, no permanent trajectory
    log_interval=25,
    out_dir="runs/muon-ab-muon",
    device="cuda",
    dtype="bfloat16",
    seed=1337,
)
