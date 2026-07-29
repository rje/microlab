"""NoPE-vs-RoPE A/B — RoPE arm (control; ~124M, GPT-2-small class: 12L x 768, RMSNorm +
SwiGLU, Muon). IDENTICAL to configs/nope-ab-nope.py except `pos` and out_dir; sizing is
the muon-ab recipe reused wholesale (batch 40 x accum 4 x block 1024, 4500 steps ~737M
tokens, ~1.7h/arm on the RTX 6000 Ada, peak ~25GB) so curves are comparable across A/Bs.

WHY THIS ABLATION: Kimi K3 ships globally-NoPE at frontier scale, and this lab just paid
heavily for RoPE context extension on the 1B (ABF tuning, retrieval decay across three
anneals — docs/sota-parity-1b.md). Kazemnejad et al. (arXiv 2305.19466) argue a decoder
with NO explicit positional encoding — position inferred from the causal mask alone —
matches explicit schemes at train length and length-generalizes BETTER. If that holds
here, the next pretrain can drop RoPE and skip the extension pain entirely. This arm is
the control: same everything, rotary position embeddings on.

After both arms finish, compare with scripts/eval_length_gen.py (val loss at 512/1024/
2048/4096 + passkey grid; the RoPE cache is extended at NATIVE theta — no ABF — because
raw extrapolation is the honest comparison) and scripts/analyze_nope_ab.py.

    python scripts/pretrain.py configs/nope-ab-rope.py --data-dir data/shards/fineweb-100bt
"""

from microlab.train.config import RunConfig

config = RunConfig(
    # model (~124M with vocab 32k; rope means no positional-embedding params — both arms
    # have byte-identical param trees, so this is a pure positional-information ablation)
    vocab_size=32000,
    block_size=1024,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.0,
    norm="rms",
    pos="rope",
    mlp="swiglu",
    # optim / schedule — lab-standard Muon (matrices) + AdamW (embeddings/norms),
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
    batch_size=40,
    grad_accum=4,
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
    out_dir="runs/nope-ab-rope",
    device="cuda",
    dtype="bfloat16",
    seed=1337,
)
