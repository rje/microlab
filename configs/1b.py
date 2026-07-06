"""~1B-parameter pretraining config — the capstone run (~1-3 weeks on an RTX 6000 Ada).
Modern block: RoPE + RMSNorm + SwiGLU. ~983M params, ~21B tokens (~20x params, Chinchilla-
optimal). Dense on purpose: MoE would store every expert (only k compute, all N resident) and
blow past 48GB. grad_checkpoint is ON — a 1B's activations overflow 48GB without it; drop
batch_size further only if the local validation still OOMs. Effective batch = batch_size *
grad_accum * block_size = 512 * 1024.

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
    compile=True,           # ~2x faster; the 350M run relied on it (missing here would crawl)
    # max-autotune: autotuned Triton kernels beat cuBLAS on our shapes. Slow one-time compile.
    compile_mode="max-autotune",
    # OFF on purpose: under torch.compile, Inductor's partitioner already minimizes activation
    # memory, so explicit grad-checkpointing is redundant AND adds a recompute tax. Off is ~30%
    # faster (18.0k vs 13.9k tok/s) at the SAME ~13GB peak. Measured; ~13.5 days for 21B tokens.
    grad_checkpoint=False,
    eval_interval=500,      # ~4h: a val-perplexity point every ~500 steps (~80 over the run)
    eval_iters=100,
    # Two-tier checkpointing (1.7TB free, ~11GB/ckpt). Rolling: every 250 steps (~2h), keep
    # the last 4 (~44GB) so a crash/reboot costs <=2h. Milestones: every 2000 steps (~1B
    # tokens), permanent — 20 checkpoints (~220GB) preserving the training trajectory for
    # later emergence/interpretability study and warm-starting ablations.
    ckpt_interval=250,
    ckpt_keep=4,
    ckpt_milestone_interval=2000,
    log_interval=20,
    out_dir="runs/1b",
    device="cuda",
    dtype="bfloat16",
    seed=1337,
)
