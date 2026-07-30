"""GDN long-run check — the GATING experiment for the hybrid lane.

At 4500 steps the arms were at parity (+0.0024 nats) but the hybrid LED early (-0.0107 at
step 1000), crossed over near step 2750, and the gap was still WIDENING at the end. That
trend, not the final number, is the reason this lane is not yet adopted: "parity at 4500
steps" may not be parity at pretrain length.

This runs 15000 steps (3.3x the tokens) to answer one question: does the gap stabilise or
keep growing? Everything else is identical to configs/gdn-ab-{dense,hybrid}.py, including the seed.

CORRECTION (2026-07-30): I originally claimed the first 4500 steps would reproduce the
4500-step runs as a free consistency check. That is WRONG and the claim is withdrawn —
lr_decay_steps also changes (15000 vs 4500), so at step 4500 this run is still mid-cosine
at a high LR while the short run had fully decayed to min_lr. Measured divergence at step
4500 is +0.082 nats, entirely accounted for by the schedule. Same seed and data order do
NOT imply the same trajectory when the LR schedule differs.

What IS valid is the comparison WITHIN this pair: gdn-long-dense vs gdn-long-hybrid share
seed, data order, and schedule, and differ only in hybrid_every. Do not cross-compare
either of them against the 4500-step arms.
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
    pos="rope",
    mlp="swiglu",
    block_norm="peri",
    hybrid_every=None,
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
    max_steps=15000,
    lr_decay_steps=15000,
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
    ckpt_interval=1000,
    ckpt_keep=2,         # ablation run: rolling recovery only, no permanent trajectory
    log_interval=25,
    out_dir="runs/gdn-long-dense",
    device="cuda",
    dtype="bfloat16",
    # seed is the SINGLE source of truth for init AND data order (both arms share it).
    # Variance measurement (finding #9's std claim needs 2-3 seeds/arm): the orchestrator
    # copies this file changing ONLY seed and out_dir (e.g. seed=1338,
    # out_dir="runs/periln-ab-peri-s1338"), then passes every run dir to
    # scripts/analyze_periln_ab.py side by side.
    seed=1337,
)
