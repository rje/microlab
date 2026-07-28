"""Stage-1 context extension REDO at gentle LR. v1 (runs/1b-4k, peak 1e-4 inherited from
the GQA-recovery schedule) damaged short-context benchmarks -2 pts by step 500 then spent the
back half re-converging — a too-hot warm restart, not ABF or data drift (milestone evals:
45.8@500 -> 46.8@1000 hellaswag). Extension itself is nearly free (passkey snapped in within
20 smoke steps), so v2 nudges: peak 3e-5, 400 steps (~0.21B tokens).
"""

from microlab.train.config import RunConfig

config = RunConfig(
    # model — identical to configs/1b.py except the context window and RoPE base
    vocab_size=32000,
    block_size=4096,
    n_layer=24,
    n_head=14,
    n_embd=1792,
    n_kv_head=None,      # ORIGINAL full-MHA model (GQA variant regressed; see docstring)
    rope_base=100000.0,  # ABF: adjusted base frequency, 10k -> 100k for the 4x window
    dropout=0.0,
    norm="rms",
    pos="rope",
    mlp="swiglu",
    # optim / schedule — see LR rationale in the module docstring
    optimizer="muon",
    muon_lr=0.004,
    lr=3e-5,
    min_lr=1e-5,
    weight_decay=0.1,
    grad_clip=1.0,
    warmup_steps=50,
    max_steps=400,
    lr_decay_steps=400,
    # data / io  (1 * 4096 * 128 = 524288 tokens/step * 1000 = 0.52B tokens)
    batch_size=1,
    grad_accum=128,
    compile=True,
    # "-no-cudagraphs" REQUIRED: plain max-autotune captures CUDA graphs, whose static
    # memory pool is incompatible with the tied lm_head/wte weight under grad
    # accumulation — crashes on the first backward (see configs/1b.py).
    compile_mode="max-autotune-no-cudagraphs",
    # OFF per the sizing measurement above: batch 1 fits in 27.7 GB reserved without the
    # ~15% recompute tax, and keeps the compile path identical to the base 1B run.
    grad_checkpoint=False,
    eval_interval=100,   # extension-curve resolution: 10 val points across the run
    eval_iters=100,
    # Two-tier checkpointing for a short run: rolling every 100 steps keep 4 (a crash
    # costs <=100 steps); milestones every 500 permanent (mid-run + final model).
    ckpt_interval=100,
    ckpt_keep=4,
    ckpt_milestone_interval=500,
    log_interval=20,
    out_dir="runs/1b-4k-v2",
    device="cuda",
    dtype="bfloat16",
    # NOT the base run's 1337 (nor the GQA uptrain's 2337): a fresh seed gives a fresh
    # data-sampling stream instead of replaying batches the base model already saw.
    seed=3337,
)
