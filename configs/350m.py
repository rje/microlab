"""~350M-parameter pretraining config (GPT-2-medium class, modern block: RoPE + RMSNorm +
SwiGLU) on FineWeb-Edu. The graduation from the TinyStories toy: real, diverse web text so
the model develops world knowledge and in-context learning (induction heads), and becomes a
legitimate post-training base. Chinchilla-optimal (~20x params ≈ 6.8B tokens).

    python scripts/pretrain.py configs/350m.py --data-dir data/shards/fineweb

Measured on the RTX 6000 (48GB): batch 16 + torch.compile peaks at ~24GB, leaving ~23GB
free — enough to serve the console Playground on GPU alongside training.
"""

from microlab.train.config import RunConfig

config = RunConfig(
    # model (~335M with vocab 32k; 24 x 1024, 16 heads — GPT-2-medium proportions)
    vocab_size=32000,
    block_size=1024,
    n_layer=24,
    n_head=16,
    n_embd=1024,
    dropout=0.0,
    norm="rms",
    pos="rope",
    mlp="swiglu",
    # optim / schedule (lower LR than the 105M run — bigger models train at lower LR;
    # matches the 1B config. muP transfer would set this principledly; 3e-4 is the safe
    # GPT-2-medium/Llama-class default until the Phase-4 muP sweep is run.)
    lr=3e-4,
    min_lr=3e-5,
    weight_decay=0.1,
    grad_clip=1.0,
    warmup_steps=700,
    max_steps=13000,
    lr_decay_steps=13000,
    # data / io  (16 * 32 * 1024 ≈ 0.52M tokens/step * 13000 ≈ 6.8B tokens ≈ 20x params,
    # ~1 epoch over the 7B prepped tokens so there's no repetition tax at all)
    batch_size=16,
    grad_accum=32,
    # compile halves memory (33.9GB -> 24.4GB) AND speeds the run; grad_checkpoint stays
    # off since compile already leaves ample headroom and checkpointing would cost ~25%.
    compile=True,
    grad_checkpoint=False,
    eval_interval=250,
    eval_iters=100,
    # ckpt every 500 (~1h of compute at risk on a crash) and keep ALL of them: the ~27
    # checkpoints (~110GB) are the substrate for the Phase-5 induction phase-change study,
    # which wants dense spacing across training. Prune later once that study is done.
    ckpt_interval=500,
    ckpt_keep=0,
    log_interval=10,
    out_dir="runs/350m",
    device="cuda",
    dtype="bfloat16",
    seed=1337,
)
