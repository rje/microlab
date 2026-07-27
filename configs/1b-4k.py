"""Context extension stage 1 (1024 -> 4096) of the ORIGINAL MHA 1B capstone: continue
training runs/1b/ckpt_40000.pt at block_size=4096 with the RoPE base raised 10k -> 100k.
NOT the GQA variant — that uptrain measured a capability regression (docs/sota-parity-1b.md),
so the extension track stays on the full-MHA base model.

    python scripts/pretrain.py configs/1b-4k.py --data-dir data/shards/fineweb-100bt \
        --init-ckpt runs/1b/ckpt_40000.pt

METHOD — adjusted base frequency (ABF), one method only: raise the RoPE base (theta)
10000 -> 100000 and train at the longer block; everything else inherits the model
unchanged. This is the Code Llama / Llama-3-style base-frequency increase: lowering all
rotation frequencies barely moves the high-frequency dims (short-range behavior is
preserved) while stretching the low-frequency dims so 4096 positions stay inside a
well-resolved rotation range. Chosen over plain positional interpolation (Chen et al.
2023, papers/architecture) because PI uniformly compresses positions — blurring
short-range resolution and baking a scale factor into inference forever — whereas ABF
with continued pretraining is the current standard for modest (4x) extensions. 10x base
for a 4x window is deliberately generous (Llama 3 used 50x for 8x).

DATA: data/shards/fineweb-100bt are flat token streams, so 4096-token training sequences
are just longer get_batch windows. Windows cross document boundaries, exactly as in base
pretraining — no long-document curation in stage 1.

LR: adaptation of a fully-trained model, not fresh pretraining — the same gentle recovery
schedule family as the GQA uptrain: muon_lr=0.004 (1/5 of the validated fresh-run 0.02)
on the block matrices, AdamW lr=1e-4 -> min_lr=1e-5 on embeddings/norms (the tied
wte/lm_head should barely move), 40x muon/adamw ratio, 5% warmup while the
long-position/raised-base gradient shock is largest, one cosine so the run ends settled.

SIZING (measured, 2026-07-27 smoke on the RTX 6000 Ada, uncompiled bf16 Muon steps at
block 4096): batch_size=1 without gradient checkpointing hit 9.7k tok/s at 27.7 GB
reserved — the fastest option under the 36 GB training budget (batch 2 no-ckpt: 10.9k
tok/s but 42.6 GB reserved, over budget; batch 4 + grad_checkpoint: 8.4k tok/s, 24.7 GB).
No grad_checkpoint also keeps the compile path identical to the validated 1b.py recipe.
batch 1 * accum 128 * 4096 = 524288 tokens/step — exactly the base run's effective-batch
convention — so 1000 steps = 0.52B tokens (~2.5% of the base run's 21B). ~15h uncompiled;
max-autotune measured ~29% faster on this model -> ~11h."""

from microlab.train.config import RunConfig

config = RunConfig(
    # model — identical to configs/1b.py except the context window and RoPE base
    vocab_size=32000,
    block_size=4096,
    n_layer=24,
    n_head=14,
    n_embd=1792,
    n_kv_head=None,      # ORIGINAL full-MHA model (GQA variant regressed; see docstring)
    rope_base=100000.0,  # ABF: adjusted base frequency, 10k -> 100k for the 4x window
    dropout=0.0,
    norm="rms",
    pos="rope",
    mlp="swiglu",
    # optim / schedule — see LR rationale in the module docstring
    optimizer="muon",
    muon_lr=0.004,
    lr=1e-4,
    min_lr=1e-5,
    weight_decay=0.1,
    grad_clip=1.0,
    warmup_steps=50,
    max_steps=1000,
    lr_decay_steps=1000,
    # data / io  (1 * 4096 * 128 = 524288 tokens/step * 1000 = 0.52B tokens)
    batch_size=1,
    grad_accum=128,
    compile=True,
    # "-no-cudagraphs" REQUIRED: plain max-autotune captures CUDA graphs, whose static
    # memory pool is incompatible with the tied lm_head/wte weight under grad
    # accumulation — crashes on the first backward (see configs/1b.py).
    compile_mode="max-autotune-no-cudagraphs",
    # OFF per the sizing measurement above: batch 1 fits in 27.7 GB reserved without the
    # ~15% recompute tax, and keeps the compile path identical to the base 1B run.
    grad_checkpoint=False,
    eval_interval=100,   # extension-curve resolution: 10 val points across the run
    eval_iters=100,
    # Two-tier checkpointing for a short run: rolling every 100 steps keep 4 (a crash
    # costs <=100 steps); milestones every 500 permanent (mid-run + final model).
    ckpt_interval=100,
    ckpt_keep=4,
    ckpt_milestone_interval=500,
    log_interval=20,
    out_dir="runs/1b-4k",
    device="cuda",
    dtype="bfloat16",
    # NOT the base run's 1337 (nor the GQA uptrain's 2337): a fresh seed gives a fresh
    # data-sampling stream instead of replaying batches the base model already saw.
    seed=3337,
)
