"""DATA LANE 1 — how much does repeating code data cost? (Muennighoff et al. repetition)

Two arms, 15000 steps each (2.46B tokens = 0.99x Chinchilla for 124M), identical in every
way EXCEPT how much unique data exists behind the same gradient budget:

  code-rep-fresh : all 274 shards, 27.35B unique -> 0.09 epochs (effectively infinite)
  code-rep-3ep   : 8 shards,        800M unique  -> 3.1 epochs

The 8 shards are chosen language-proportionally (3 Python : 3 JavaScript : 2 TypeScript,
matching the corpus's 10:10:7.35 token split), so the arms differ ONLY in unique-data
volume and not in language mix. Validation is the SAME held-out shard for both, routed by
content hash at build time and therefore disjoint from every train shard.

WHY THIS LANE FIRST. It is the data question our own corpus build forced on us: TypeScript
EXHAUSTED the permissively-licensed Stack at 7.38B tokens, short of its 10B budget. If
repetition is near-free up to ~4 epochs (the published finding), that ceiling never binds
and we can scale the specialist without new data agreements. If repetition is costly, the
TS ceiling is a hard constraint on the design and we need to know before Phase B, not
after. It also needs zero new tooling — no FIM transform, no retokenised FineWeb — which
the general-first and mix lanes both do.

Evaluated on val loss. Not on HumanEval: 124M sits at the measured 0.0% floor there, so
downstream code metrics cannot discriminate at this scale.
"""

from microlab.train.config import RunConfig

config = RunConfig(
    # model (~124M with vocab 32k; Peri-LN block layout — see formulation note above)
    vocab_size=49152,
    block_size=1024,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.0,
    norm="rms",
    pos="rope",
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
    out_dir="runs/code-rep-fresh",
    device="cuda",
    dtype="bfloat16",
    # seed is the SINGLE source of truth for init AND data order (both arms share it).
    # Variance measurement (finding #9's std claim needs 2-3 seeds/arm): the orchestrator
    # copies this file changing ONLY seed and out_dir (e.g. seed=1338,
    # out_dir="runs/periln-ab-peri-s1338"), then passes every run dir to
    # scripts/analyze_periln_ab.py side by side.
    seed=1337,
)
