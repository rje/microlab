"""SMOKE TEST of the 4x cloud path — NOT the capstone run.

60 steps, checkpoint every 20, compile off. It exists to answer the two things only real
multi-GPU hardware can: does NCCL come up across 4 ranks, and does throughput actually
scale? The $159 projection for the full run rests on an ASSUMED 96% scaling efficiency
that has never been measured.

It also exercises the whole preemptible path end to end: bid provisioning, corpus pull,
checkpoint upload to B2, and clean destroy. Uses run-prefix `smoke-4x` so it cannot
collide with the real run's checkpoints.

Original header follows.

Microlab-Coder-1B — the capstone pretrain. Frontier stack at 1B, 32k context.

Architecture is `configs/frontier-32k.py` scaled up; that run validated the stack
end to end (15,000 steps, val 1.0611, no instability, decode verified). Deviations from
Kimi Linear remain exactly two: 1B dense rather than 48B-A3B MoE, and the code-49k
tokenizer (a fertility measurement on our corpus, not a taste call).

SHAPE: 24 x 1792, 14 heads, head_dim 128. head_dim 128 is universal in the 1-3B cohort and
it is also the recall lever — the KDA state is n_head * head_dim^2 = 229,376 values/layer,
4.67x the 124M's. See docs/small-model-long-retrieval-lit.md.

WHAT THIS RUN CLAIMS. A compute-optimal code model with a 32k WINDOW. It does NOT claim
retrieval across that window, and the plan must not be gated on it:
  - our own 1B needed ~17B tokens (80% of Chinchilla) before it could retrieve across a
    1,024-token window at all — measured, not assumed (0.04 at 8.4B tokens, 0.87 at 16.8B);
  - RULER shows 7B+ models mostly failing at 32k;
  - Zoology/Based show recall is state-bound and linear-attention models are the weak class.
Retrieval will be MEASURED at milestone checkpoints and REPORTED. If it emerges, good; it
is not a success criterion, because nothing in the literature says it should at this budget.

MIGRATION IS A FIRST-CLASS CONSTRAINT. This run starts locally (~27 days) with the
intention of possibly finishing on 8xH100. That only works if the global batch is invariant
to world size, so the batch is specified in TOKENS PER STEP and grad_accum is derived:

    grad_accum = tokens_per_step / (world_size * batch_size * block_size)

  local  (1 GPU):  524288 / (1 * 1 * 32768) = 16
  8xH100 (8 GPU):  524288 / (8 * 1 * 32768) = 2

Same optimizer trajectory either side. Change grad_accum by hand instead and the effective
batch moves 8x mid-run, silently invalidating the LR schedule and the token accounting.
524,288 tokens/step also matches the original 1B capstone, so its loss curve is a reference.

DURATION: 40,000 steps x 524,288 = 21.0B tokens = 20.2x params, Chinchilla-optimal and the
same budget the first 1B used.
"""

from microlab.train.config import RunConfig

config = RunConfig(
    # model — ~1.04B params
    vocab_size=49280,   # 49,152 base + 3 FIM sentinels, padded to 385x128
    block_size=32768,
    n_layer=24,
    n_head=14,
    n_embd=1792,          # head_dim = 1792/14 = 128
    dropout=0.0,
    norm="rms",
    pos="nope",
    mlp="swiglu",
    block_norm="peri",
    hybrid_every=4,       # 18 KDA : 6 MLA, the Kimi Linear 3:1 ratio
    gdn_gate="channel",   # KDA per-channel gate, not GDN's scalar
    global_attn="mla",
    mla_kv_lora=512,
    qk_norm=True,
    # optim — Muon on matrices, AdamW on embeddings/norms. LR scaled down from the 124M's
    # 6e-4/0.02: the original 1B capstone used 3e-4 AdamW at this width and was stable.
    optimizer="muon",
    muon_lr=0.01,
    lr=3e-4,
    min_lr=3e-5,
    weight_decay=0.1,
    grad_clip=1.0,
    warmup_steps=5,     # same fraction of the run as the first 1B (700/40000)
    max_steps=60,
    lr_decay_steps=60,
    # memory — both REQUIRED at 32k on a 48GB card; the gate hard-fails without them.
    # Measured at 1B: 32768-token steps cost 27.70 GB naive, 15.40 GB with fused CE.
    grad_checkpoint=True,
    fused_ce=True,
    batch_size=1,
    # tokens_per_step is AUTHORITATIVE; grad_accum is DERIVED per world size
    # (16 at ws=1, 4 at ws=4, 2 at ws=8) so the optimizer sees the same batch either way.
    tokens_per_step=524288,
    grad_accum=16,        # fallback only, when tokens_per_step is 0
    compile=False,
    compile_mode="max-autotune-no-cudagraphs",
    eval_interval=1000,
    eval_iters=40,
    # Checkpoints sized for a run that may migrate: rolling recovery plus a permanent
    # trajectory. The milestones are what the emergence sweep reads — measuring WHEN
    # retrieval appears is only possible if the trajectory is kept.
    ckpt_interval=20,
    ckpt_keep=2,
    ckpt_milestone_interval=0,
    log_interval=5,
    data_dir="data/shards/mix-v1",   # NOT BUILT YET — blocking, see the plan
    out_dir="runs/coder-1b",
    device="cuda",
    dtype="bfloat16",
    seed=1337,
)
