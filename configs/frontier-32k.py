"""FRONTIER stack — the Kimi Linear pattern at 1B-class shape, on repo-packed 32k code.

Basis: Kimi Linear (3:1 KDA:MLA, NoPE globals). Owner's governance call after a week in
which my "acceptable" judgements were wrong repeatedly: take the shipped frontier design as
the DEFAULT and require evidence to deviate, not to conform.

DEVIATIONS FROM KIMI LINEAR — the complete list, both scale-driven rather than taste:
  1. 1B dense, not 48B-A3B MoE. Our own depth/width follows.
  2. code-49k tokenizer — a fertility MEASUREMENT on our corpus, not a judgement.

Components, all tested (801 tests green):
  - KDA linear layers, per-channel DPLR gate, fused Triton kernel (flash-linear-attention),
    equivalence-gated against our float64 reference at the bf16 floor.
  - MLA global layers: latent KV, per-head distinct K/V. With NoPE there is no decoupled
    RoPE, so the cache is exactly mla_kv_lora = 512 values/token — identical to GQA(2),
    which caches the same 512 but shares one K/V head across 7 queries.
  - NoPE: NO positional encoding anywhere in the model. No RoPE table, no theta, no ABF,
    no YaRN. The context-extension apparatus does not apply.
  - QK-norm, head_dim variant (Qwen3/Gemma-3, not OLMo 2's full-projection form).
  - Fused linear+cross-entropy (Liger): 40-44% off training memory, validated against fp32
    truth rather than against our own bf16 path.
  - Peri-LN, SwiGLU, tied embeddings, Muon.

DATA: data/shards/code-repo-32k — repo-level packed. On the file-level corpus a 32k window
spanned 19.6 UNRELATED documents (median doc 664 tokens); packed it spans 0.7, i.e. less
than one repository. Long context is meaningless without this.

CONTEXT 32768 is the measured choice, not a guess: with fused kernels the hybrid is 0.90x
dense at 32k (it was 5.8x SLOWER unfused at 8k), so long context costs nothing relative to
the alternative.

THIS IS A VALIDATION RUN, NOT A PRETRAIN. It answers: does the full stack train stably at
32k, does throughput match the benchmark, and can the model USE its context (passkey,
length generalisation)? It deliberately does NOT try to prove a marginal loss win — our
124M A/Bs have produced -0.0020 and -0.0059 nats, both noise-adjacent, one of which
inverted on duration.
"""

from microlab.train.config import RunConfig

config = RunConfig(
    # model (~124M with vocab 32k; Peri-LN block layout — see formulation note above)
    vocab_size=49152,
    block_size=32768,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.0,
    norm="rms",
    pos="nope",
    mlp="swiglu",
    block_norm="peri",
    hybrid_every=4,
    gdn_gate="channel",
    global_attn="mla",
    mla_kv_lora=512,
    qk_norm=True,
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
    batch_size=1,
    grad_accum=8,
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
    data_dir="data/shards/code-repo-32k",
    out_dir="runs/frontier-32k",
    device="cuda",
    dtype="bfloat16",
    # seed is the SINGLE source of truth for init AND data order (both arms share it).
    # Variance measurement (finding #9's std claim needs 2-3 seeds/arm): the orchestrator
    # copies this file changing ONLY seed and out_dir (e.g. seed=1338,
    # out_dir="runs/periln-ab-peri-s1338"), then passes every run dir to
    # scripts/analyze_periln_ab.py side by side.
    seed=1337,
)
