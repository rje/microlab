"""~1B-parameter pretraining config — the capstone run (~1-3 weeks on an RTX 6000 Ada).
Modern block: RoPE + RMSNorm + SwiGLU. ~983M params, ~21B tokens (~20x params, Chinchilla-
optimal). Dense on purpose: MoE would store every expert (only k compute, all N resident) and
blow past 48GB. batch_size=8 keeps the activation stack under budget without CUDA graphs (see
compile_mode). Effective batch = batch_size * grad_accum * block_size = 512 * 1024.

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
    # data / io  (8 * 1024 * 64 ≈ 0.52M tokens/step * 40000 ≈ 21B tokens)
    batch_size=8,
    grad_accum=64,
    compile=True,           # ~2x faster; the 350M run relied on it (missing here would crawl)
    # Autotuned Triton kernels beat cuBLAS on our shapes (slow one-time compile). The
    # "-no-cudagraphs" is REQUIRED: plain max-autotune captures CUDA graphs, whose static
    # memory pool is incompatible with our tied lm_head/wte weight under grad accumulation
    # (multiple forward/backward per step) — it crashes on the first backward. Dropping only
    # cudagraphs keeps the Triton autotuning win.
    compile_mode="max-autotune-no-cudagraphs",
    # OFF: at batch_size=8 the activation stack fits (~34GB peak) without CUDA graphs, so we
    # skip the recompute tax and run faster (17.9k vs 14.5k tok/s -> ~13.6 vs ~16.7 days). The
    # only way to keep batch 16 would be cudagraphs (crashes on tied weights, see compile_mode)
    # or grad_checkpoint=True; batch 8 + off is both faster AND the same effective batch (512).
    # Measured on the real Trainer step.
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
