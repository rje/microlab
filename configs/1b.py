"""~1B-parameter pretraining config — the capstone run (~1-3 weeks on an RTX 6000 Ada).
Modern block: RoPE + RMSNorm + SwiGLU. ~20B tokens (~20x params). Requires the ~150M run
to have validated the pipeline first. Enable gradient checkpointing / lower batch_size if
you OOM; the effective batch is batch_size * grad_accum * block_size.

    python scripts/pretrain.py configs/1b.py    # resumable across interruptions
"""

from microlab.train.config import RunConfig

config = RunConfig(
    # model (~1.0B with vocab 32k)
    vocab_size=32000,
    block_size=1024,
    n_layer=24,
    n_head=14,
    n_embd=1792,
    dropout=0.0,
    norm="rms",
    pos="rope",
    mlp="swiglu",
    # optim / schedule
    lr=3e-4,
    min_lr=3e-5,
    weight_decay=0.1,
    grad_clip=1.0,
    warmup_steps=700,
    max_steps=40000,
    lr_decay_steps=40000,
    # data / io  (16 * 1024 * 32 ≈ 0.52M tokens/step * 40000 ≈ 21B tokens)
    batch_size=16,
    grad_accum=32,
    eval_interval=1000,
    eval_iters=200,
    ckpt_interval=1000,
    log_interval=20,
    out_dir="runs/1b",
    device="cuda",
    dtype="bfloat16",
    seed=1337,
)
