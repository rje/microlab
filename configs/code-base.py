"""Code-specialist data-lane BASE config — 124M on the-stack code corpus.

The architecture is FROZEN to the adopted recipe so these lanes vary DATA only:
Peri-LN + Muon + RoPE + SwiGLU + the 3:1 GDN hybrid (verdict 3, adopted 2026-07-30).

Differences from the FineWeb ablation configs, all forced by the corpus:
  - vocab_size 49152 (code-49k tokenizer, ~44% fewer tokens/byte on code) not 32000
  - data-dir data/shards/code-stack-30b (27.35B train / 28.7M val tokens)

Duration is 15000 steps = 2.46B tokens = 0.99x Chinchilla for 124M, per the protocol change
of 2026-07-30: a 4500-step lane is 0.30x Chinchilla and can invert its own verdict.

PERI-LN NOTE (resolved 2026-07-31): the retest at 15000 steps shrank Peri-LN's effect from
-0.0152 to -0.0020 nats, below the 0.0025 paired band — kept because it is free, but it is
not a quality win. Immaterial to these lanes either way: all data arms share the
architecture, so the RELATIVE data comparison is unaffected.

!! THIS IS AN ABLATION CONFIG, NOT A PRETRAIN CONFIG. !!
Its block_size (1024), absent n_kv_head (= full MHA on the 3 global layers) and absent
rope_base (= 10000) are ALL THREE of the divergences docs/sota-parity-1b.md flagged as
must-fix on the 1B. They are acceptable here — short-context ablations are cheap and the
data lanes only need arms to be mutually comparable — and they are RECORDED here precisely
so they cannot ride along into a pretrain config the way they did last time (parity review
finding #8: defaults that never surfaced as questions).

Two are load-bearing beyond parity:
- block_size 1024 exercises almost NONE of what the GDN hybrid was adopted for (4x KV
  reduction, ~10x better length generalisation, latency crossover at ~100k). A data lane
  measured here is still valid for DATA questions; do not read it as validating the
  architecture at length.
- full MHA on the global layers is the exact error that cost the 1B an 8x KV cache.

Before any pretrain: run the parity review (docs/sota-parity-code-specialist.md) and set
n_kv_head, block_size and rope_base deliberately.

Shard->language map (verified by decoding samples, 100M tokens per shard):
  train-00000..00099 Python | 00100..00199 JavaScript | 00200..00273 TypeScript
so language-mix arms are made by subsetting the manifest, not by rebuilding the corpus.
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
    out_dir="runs/code-base",
    device="cuda",
    dtype="bfloat16",
    # seed is the SINGLE source of truth for init AND data order (both arms share it).
    # Variance measurement (finding #9's std claim needs 2-3 seeds/arm): the orchestrator
    # copies this file changing ONLY seed and out_dir (e.g. seed=1338,
    # out_dir="runs/periln-ab-peri-s1338"), then passes every run dir to
    # scripts/analyze_periln_ab.py side by side.
    seed=1337,
)
