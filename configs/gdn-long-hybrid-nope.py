"""NoPE-on-globals RETESTED at compute-optimal — because the first verdict violated our
own duration rule and disagrees with a shipped frontier design.

Identical to configs/gdn-long-hybrid.py except pos="nope". Compares directly against
runs/gdn-long-hybrid (RoPE-globals, 15000 steps, same seed) which is already trained.

WHY. The NoPE-globals verdict in docs/gdn-hybrid-verdict.md ("keep RoPE, it beats NoPE 5x
on length generalisation") was measured on ckpt_4500 = 0.34x Chinchilla — the exact regime
that INVERTED the GDN hybrid verdict. It was recorded as settled anyway. Kimi Linear ships
NoPE on its global layers at frontier scale, so a confident contrary result from an
under-trained 124M run deserves the retest before it stands.

Two other reasons our NoPE arm may be handicapped relative to Kimi's, NOT fixed by this run
and recorded so the result is read correctly:
  1. GATING CAPACITY. Our decay gate is nn.Linear(n_embd -> n_head): one scalar per head.
     Kimi's KDA uses a Diagonal-Plus-Low-Rank transition, i.e. per-channel gating
     (n_embd -> n_embd) — ~64x more capacity. If NoPE works because the RECURRENCE carries
     position, a scalar decay carries far less of it, and our NoPE arm is weakened in
     precisely the way that would produce our result.
  2. DEPTH. Haviv et al. locate NoPE's positional signal in middle layers (we replicated:
     MAD 36.8 mid-stack). We have 12 layers, only 3 of them attention. Kimi is far deeper.

So this run tests hypothesis (0) — under-training — which is the cheapest of the three. If
NoPE-globals still loses at 15000 steps, hypotheses (1) and (2) remain live and the honest
statement is "NoPE-globals loses AT OUR SCALE AND WITH OUR GATE", not "Kimi is wrong".
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
    out_dir="runs/gdn-long-hybrid-nope",
    device="cuda",
    dtype="bfloat16",
    # seed is the SINGLE source of truth for init AND data order (both arms share it).
    # Variance measurement (finding #9's std claim needs 2-3 seeds/arm): the orchestrator
    # copies this file changing ONLY seed and out_dir (e.g. seed=1338,
    # out_dir="runs/periln-ab-peri-s1338"), then passes every run dir to
    # scripts/analyze_periln_ab.py side by side.
    seed=1337,
)
