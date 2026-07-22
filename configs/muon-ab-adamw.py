"""Muon A/B validation — AdamW arm (~124M params, GPT-2-small class: 12L x 768, RoPE +
RMSNorm + SwiGLU). IDENTICAL to configs/muon-ab-muon.py except `optimizer` and out_dir;
compare val-loss curves at matched steps (evals use the same seeded batches in both arms).

Sizing (measured on the RTX 6000 Ada, compile=max-autotune-no-cudagraphs, batch 40 x
accum 4 x block 1024 = 163,840 tok/step): adamw 1.27 s/step / muon 1.32 s/step, so 4500
steps ~= 1.6-1.7h per arm, ~737M tokens (~6x params). Peak mem ~25GB.

    python scripts/pretrain.py configs/muon-ab-adamw.py --data-dir data/shards/fineweb-100bt
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
    # optim / schedule — repo's 150M AdamW conventions
    optimizer="adamw",
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
    out_dir="runs/muon-ab-adamw",
    device="cuda",
    dtype="bfloat16",
    seed=1337,
)
