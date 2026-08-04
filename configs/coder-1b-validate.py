"""PIPELINE VALIDATION arm of coder-1b. Identical architecture, data and optimiser —
the ONLY change is ckpt_interval 50 -> 25.

Not a training config. It exists to prove the paid path end to end (stream -> train ->
checkpoint -> B2 -> preempt -> resume) inside a single short rental, which the production
interval cannot do: at ~54 s/step the first checkpoint at step 50 is 45 minutes of
training, longer than the uptime windows actually observed on interruptible hosts, so
every run so far died before writing anything durable. 25 halves the time-to-first-proof.

Run it under its OWN --run-prefix so validation checkpoints never land in the real run's
namespace. 25 is deliberately NOT promoted to the production config: at 40,000 steps it
would spend roughly 20 hours serialising checkpoints.
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
    # COMPILE OFF on rented hardware, and the reason is narrower than I twice claimed.
    # Liger's fused CE calls addmm with an out_dtype kwarg that dynamo cannot trace in some
    # builds:
    #   addmm(..., out_dtype=torch.float32, out=FakeTensor(...))
    #   TypeError: unsupported operand type(s) for *: 'torch.dtype' and 'FakeTensor'
    # It is NOT a torch 2.11 bug (my second guess): it reproduces on 2.12.1+cu126, the same
    # VERSION that runs frontier-32k fine locally on 2.12.1+cu130. The variable is the CUDA
    # build, not the release. Dropping fused CE instead is not available — it is what makes
    # 32k fit at all (27.70 -> 15.40 GB measured).
    # Compile's benefit here is also UNMEASURED and plausibly small: it was worth ~0 on the
    # RTX 6000 Ada at 32k. Treat it as a separate experiment rather than a blocker.
    compile=False,
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
    ckpt_interval=25,
    ckpt_keep=3,
    ckpt_milestone_interval=2000,
    log_interval=10,
    # 15 min against a 25-39 s steady-state step. The bound is set by the SLOWEST
    # legitimate step, not the typical one: early on, a step can need up to 16 uncached
    # shards, and at the measured ~28 s each that is ~7.5 min of honest work. 900 s clears
    # that with margin while catching a wedge in minutes instead of never.
    step_timeout_s=900,
    data_dir="data/shards/mix-v1",   # NOT BUILT YET — blocking, see the plan
    out_dir="runs/coder-1b",
    device="cuda",
    dtype="bfloat16",
    seed=1337,
)
