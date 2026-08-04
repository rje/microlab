"""Microlab-Coder-1B — the capstone pretrain. Frontier stack at 1B, 32k context.

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
    warmup_steps=700,     # same fraction of the run as the first 1B (700/40000)
    max_steps=40000,
    lr_decay_steps=40000,
    # memory — both REQUIRED at 32k on a 48GB card; the gate hard-fails without them.
    # Measured at 1B: 32768-token steps cost 27.70 GB naive, 15.40 GB with fused CE.
    grad_checkpoint=True,
    fused_ce=True,
    batch_size=1,
    # tokens_per_step is AUTHORITATIVE; grad_accum is DERIVED per world size
    # (16 at ws=1, 4 at ws=4, 2 at ws=8) so the optimizer sees the same batch either way.
    tokens_per_step=524288,
    grad_accum=16,        # fallback only, when tokens_per_step is 0
    # Compile ON with autotuning. It measured ~nothing on the RTX 6000 Ada at 32k, but
    # that is a different card and the figure was never re-checked on an H100 — the same
    # class of error as assuming it gives 2x. The compile cost (~15-35 min once) is
    # negligible against a multi-day run, and it is re-paid on every preemption, which is
    # an argument for the pre-built image rather than for disabling it.
    # COMPILE ON, PER-BLOCK. The saga, because this flag has now been wrong in both
    # directions:
    #   * whole-model compile crashes on cu126 — Liger fused-CE's addmm(out_dtype=...) is
    #     untraceable there (fine on cu130). Fused CE cannot be dropped; it is what makes
    #     32k fit at all (27.70 -> 15.40 GB measured). That killed two paid attempts.
    #   * "worth ~0 at 32k" (this file, previously) described WHOLE-MODEL compile and was
    #     wrong for per-block; "~29%" (config.py, previously) was TF32+fused-AdamW+
    #     max-autotune combined on the prose 1B at 1024 ctx, not compile alone.
    # compile_scope="blocks" (the trainer default) compiles each transformer block in
    # place and keeps the loss head out of every graph, so the cu126 op never gets traced.
    # MEASURED end-to-end on this exact config (RTX 6000 Ada, 32k, grad-ckpt on):
    # 4,821 -> 7,032 tok/s = 1.46x, identical loss trajectory to 3 decimals, state_dict
    # keys unchanged. H100 number TBD on the next paid validation run.
    compile=True,
    compile_mode="max-autotune-no-cudagraphs",
    eval_interval=500,
    eval_iters=40,
    # Checkpoints sized for a run that may migrate: rolling recovery plus a permanent
    # trajectory. The milestones are what the emergence sweep reads — measuring WHEN
    # retrieval appears is only possible if the trajectory is kept.
    #
    # 250 was wrong for INTERRUPTIBLE hardware, and was set from habit rather than
    # measurement. At the observed 25-39 s/step it put the first checkpoint over 100
    # minutes out, so a preemption could discard an entire paid hour and a half — exactly
    # the loss the supervisor exists to prevent — and a capped run could exhaust its
    # budget without ever writing durable progress. 50 caps that exposure at ~30 min
    # against a ~3 min upload for a 12 GB checkpoint at measured B2 rates.
    ckpt_interval=50,
    ckpt_keep=3,
    ckpt_milestone_interval=2000,
    log_interval=10,
    # 15 min against a 25-39 s steady-state step. The bound is set by the SLOWEST
    # legitimate step, not the typical one: early on, a step can need up to 16 uncached
    # shards, and at the measured ~28 s each that is ~7.5 min of honest work. 900 s clears
    # that with margin while catching a wedge in minutes instead of never.
    step_timeout_s=900,
    data_dir="data/shards/mix-v2",   # v2: mixed val from held-out splits, chunk-level FIM
    out_dir="runs/coder-1b",
    device="cuda",
    dtype="bfloat16",
    seed=1337,
)
