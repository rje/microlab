"""~150M-parameter pretraining config (GPT-2-small class, modern block: RoPE + RMSNorm +
SwiGLU). The validation gate before the 1B capstone. ~3B tokens (~20x params) at ~0.5M
tokens/step. Tune batch_size/grad_accum to your VRAM.

    python scripts/pretrain.py configs/150m.py
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
    # optim / schedule
    lr=6e-4,
    min_lr=6e-5,
    weight_decay=0.1,
    grad_clip=1.0,
    warmup_steps=200,
    max_steps=6000,
    lr_decay_steps=6000,
    # data / io  (40 * 1024 * 12 ≈ 0.5M tokens/step * 6000 ≈ 3B tokens)
    # batch 40 uses ~38GB of the 48GB card (vs ~24GB at batch 24) — fewer micro-steps,
    # same effective batch, faster wall-clock. Drop to 24/accum 20 if VRAM is tight.
    batch_size=40,
    grad_accum=12,
    eval_interval=250,
    eval_iters=100,
    ckpt_interval=500,
    log_interval=20,
    out_dir="runs/150m",
    device="cuda",
    dtype="bfloat16",
    seed=1337,
)
