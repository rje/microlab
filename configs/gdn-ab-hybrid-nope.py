"""GDN hybrid with NoPE on the global layers — the ACTUAL Kimi Linear configuration,
and the retest of the conditional left open by docs/nope-verdict-audit.md.

Identical to configs/gdn-ab-hybrid.py except pos: "nope" instead of "rope". Because
`pos` applies only to the surviving global-attention layers (GatedDeltaNet carries
position in its recurrence), this makes the 3 global layers positionless and delegates
ALL positional modelling to the 9 linear layers. That is what Kimi Linear ships.

WHY THIS IS NOT A CONTRADICTION of verdict 1 (RoPE, pure NoPE rejected). That verdict was
about NoPE in a FULLY dense stack, where nothing else encodes position and the model must
infer it from the causal mask alone — it cost +0.057 nats and collapsed on length
generalisation. The open conditional was always whether NoPE becomes viable when
recurrence supplies position. Compare against runs/gdn-ab-hybrid (same seed, same
everything else): if this arm matches it, position does not need to be explicit in the
global layers.
"""

from microlab.train.config import RunConfig

config = RunConfig(
    # model (~124M with vocab 32k; Peri-LN block layout — see formulation note above)
    vocab_size=32000,
    block_size=1024,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.0,
    norm="rms",
    pos="nope",
    mlp="swiglu",
    block_norm="peri",
    hybrid_every=4,
    # optim / schedule — lab-standard Muon (matrices) + AdamW (embeddings/norms; the
    # extra peri norm scales are 1-D so they land on AdamW automatically),
    # values straight from the muon-ab arms
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
    batch_size=10,
    grad_accum=16,
    compile=True,
    # "-no-cudagraphs" is REQUIRED: plain max-autotune captures CUDA graphs, whose static
    # memory pool is incompatible with the tied lm_head/wte weight under grad accumulation
    # — it crashes on the first backward (see configs/1b.py).
    compile_mode="max-autotune-no-cudagraphs",
    eval_interval=250,   # 18 matched-step val-loss points per arm (same seeded batches)
    eval_iters=50,       # 50 x 40 x 1024 ~= 2M val tokens per point
    ckpt_interval=500,
    ckpt_keep=2,         # ablation run: rolling recovery only, no permanent trajectory
    log_interval=25,
    out_dir="runs/gdn-ab-hybrid-nope",
    device="cuda",
    dtype="bfloat16",
    # seed is the SINGLE source of truth for init AND data order (both arms share it).
    # Variance measurement (finding #9's std claim needs 2-3 seeds/arm): the orchestrator
    # copies this file changing ONLY seed and out_dir (e.g. seed=1338,
    # out_dir="runs/periln-ab-peri-s1338"), then passes every run dir to
    # scripts/analyze_periln_ab.py side by side.
    seed=1337,
)
