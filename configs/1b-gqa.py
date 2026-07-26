"""GQA uptrain of the 1B capstone (Ainslie et al., 2023): recover the mean-pooled
14-query/2-KV-head conversion of runs/1b/ckpt_40000.pt. Same geometry as configs/1b.py,
same data recipe, ~2000 steps = ~1.05B tokens — alpha ~= 0.05 of the base run's 21B,
exactly the paper's uptraining proportion. This model is the base for the subsequent
context-extension stage (rope_base stays 10000 here; raising it is that stage's job).

    python scripts/convert_gqa.py runs/1b/ckpt_40000.pt runs/1b-gqa-convert/converted.pt \
        --n-kv-head 2 --kl-data-dir data/shards/fineweb-100bt --device cuda
    python scripts/pretrain.py configs/1b-gqa.py --data-dir data/shards/fineweb-100bt \
        --init-ckpt runs/1b-gqa-convert/converted.pt

LR: this is RECOVERY of an already-trained model, not fresh pretraining — only the
pooled kv_proj is far from a good optimum; everything else needs mild re-adaptation. So
the schedule is gentler than the fresh-run peaks: muon_lr=0.004 (1/5 of the validated
fresh-run 0.02 from configs/muon-ab-muon.py) on the block matrices, AdamW lr=1e-4 (1/3
of the base 1B's 3e-4 peak) on embeddings/norms — the tied wte/lm_head is fully trained
and should barely move. The muon/adamw ratio (40x) stays in family with the fresh-run
convention (~33x). Short 100-step warmup (5% of the run) lets Muon's momentum buffers
fill while the pooled-KV gradient shock is largest, then one cosine to min_lr=1e-5 so
the run ends settled, as the base run did (it finished at 3e-5)."""

from microlab.train.config import RunConfig

config = RunConfig(
    # model — identical to configs/1b.py except grouped-query attention 14:2
    vocab_size=32000,
    block_size=1024,
    n_layer=24,
    n_head=14,
    n_embd=1792,
    n_kv_head=2,        # 7 query heads per KV head; 7x smaller KV cache
    rope_base=10000.0,  # unchanged; the context-extension stage raises it
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
    warmup_steps=100,
    # 2000 (the Ainslie ~5%-of-pretrain guideline) recovered val ppl 15.8 vs the base's
    # 12.19 with benchmarks down 5.5 pts mean — most of the residual gap is unrecovered
    # damage, not the 850M capacity delta (~0.03-0.05 nats), and the curve was still falling.
    # Top-up: +2500 steps (~1.3B more tokens) at the schedule's min_lr floor (lr_decay_steps
    # stays 2000, so steps 2000-4500 run at a constant 1e-5 — a standard continued-anneal).
    max_steps=4500,
    lr_decay_steps=2000,
    # data / io  (8 * 1024 * 64 ~= 0.52M tokens/step * 2000 ~= 1.05B tokens)
    batch_size=8,
    grad_accum=64,
    compile=True,
    # "-no-cudagraphs" REQUIRED: plain max-autotune captures CUDA graphs, whose static
    # memory pool is incompatible with the tied lm_head/wte weight under grad
    # accumulation — crashes on the first backward (see configs/1b.py).
    compile_mode="max-autotune-no-cudagraphs",
    # OFF per the measured 1b.py precedent: batch 8 fit at ~34GB peak with full MHA;
    # GQA strictly shrinks params and attention activations, so it fits with room.
    grad_checkpoint=False,
    eval_interval=100,   # recovery-curve resolution: 20 val points across the uptrain
    eval_iters=100,
    # Two-tier checkpointing for a short run (~10GB/ckpt): rolling every 100 steps keep 4
    # (crash costs <=100 steps); milestones every 500 permanent — 4 recovery-trajectory
    # snapshots + the final model.
    ckpt_interval=100,
    ckpt_keep=4,
    ckpt_milestone_interval=500,
    log_interval=20,
    out_dir="runs/1b-gqa",
    device="cuda",
    dtype="bfloat16",
    # NOT the base run's 1337: a fresh seed gives a fresh data-sampling stream instead of
    # replaying the exact batch sequence the base model already saw at the start of its run.
    seed=2337,
)
