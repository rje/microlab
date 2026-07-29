"""NoPE-verdict audit — HP-fairness LR sweep, x1 arm (muon_lr 0.02, the original value).

IDENTICAL to configs/nope-ab-nope.py except max_steps=1000 and out_dir: lr_decay_steps
stays 4500 so the warmup+cosine SCHEDULE is the original's, truncated not squeezed —
this arm replays the original run's first 1000 steps at the same seed and therefore
doubles as a run-to-run stability point (original step-1000 val loss: 3.9033; a
deviation > 0.02 flags single-seed noise as material to the +0.057 verdict gap).

    python scripts/pretrain.py configs/nope-audit-lr10.py --data-dir data/shards/fineweb-100bt

Sweep siblings: nope-audit-lr05.py (x0.5), nope-audit-lr20.py (x2). Compare matched-step
val losses vs runs/nope-ab-rope's first-1000-step curve (scripts/analyze_nope_ab.py
machinery / TB events): if no LR materially closes the gap trajectory, RoPE-tuned
hyperparameters are ruled out as the cause of NoPE's in-window penalty.
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
    muon_lr=0.02,
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
    out_dir="runs/nope-audit-lr10",
    device="cuda",
    dtype="bfloat16",
    seed=1337,
)
