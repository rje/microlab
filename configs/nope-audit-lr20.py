"""NoPE-verdict audit — HP-fairness LR sweep, x2 arm (muon_lr 0.04 vs original 0.02).

IDENTICAL to configs/nope-ab-nope.py except muon_lr, max_steps=1000, and out_dir:
lr_decay_steps stays 4500 so the schedule SHAPE matches the original's first 1000 steps
(truncated, not squeezed) and matched-step losses are directly comparable. Tests whether
the +0.057 in-window NoPE penalty is an artifact of RoPE-tuned Muon LR (doubling it).

    python scripts/pretrain.py configs/nope-audit-lr20.py --data-dir data/shards/fineweb-100bt

Sweep siblings: nope-audit-lr10.py (x1, also the seed-stability point),
nope-audit-lr05.py (x0.5). See nope-audit-lr10.py for the comparison protocol.
"""

from microlab.train.config import RunConfig

config = RunConfig(
    vocab_size=32000,
    block_size=1024,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.0,
    norm="rms",
    pos="nope",
    mlp="swiglu",
    optimizer="muon",
    muon_lr=0.04,
    lr=6e-4,
    min_lr=6e-5,
    weight_decay=0.1,
    grad_clip=1.0,
    warmup_steps=150,
    max_steps=1000,
    lr_decay_steps=4500,
    batch_size=40,
    grad_accum=4,
    compile=True,
    compile_mode="max-autotune-no-cudagraphs",
    eval_interval=250,
    eval_iters=50,
    ckpt_interval=500,
    ckpt_keep=2,
    log_interval=25,
    out_dir="runs/nope-audit-lr20",
    device="cuda",
    dtype="bfloat16",
    seed=1337,
)
