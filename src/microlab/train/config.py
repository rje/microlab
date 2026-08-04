"""Run configuration for the production Trainer (Phase-2 real-scale). One dataclass so a
150M and a 1B run are just different config values."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunConfig:
    # model (VariantConfig fields)
    vocab_size: int = 32000
    block_size: int = 512
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    norm: str = "rms"
    pos: str = "rope"
    mlp: str = "swiglu"
    # Block layout: "pre" -> the pre-norm block (unchanged default); "peri" -> Peri-LN,
    # y = x + Norm(Module(Norm(x))) (arXiv 2502.02732; Gemma 2/3's pre+post sandwich).
    # Old checkpoints unpickle without this attribute and fall back to this default.
    block_norm: str = "pre"
    # None -> classic multi-head attention, exactly the pre-field behavior. A divisor of
    # n_head -> grouped-query attention (n_kv_head K/V heads shared across query groups).
    # Old checkpoints unpickle without this attribute and fall back to this class default.
    n_kv_head: int | None = None
    # RoPE frequency base (theta); 10000.0 was hard-coded before this field existed.
    # Groundwork for the context-extension stage (PI/YaRN want a raised base).
    rope_base: float = 10000.0
    # None -> every layer is global attention (unchanged default). N -> Kimi-Linear-style
    # layerwise hybrid: every Nth layer stays global attention, the other N-1 become
    # GatedDeltaNet. N=4 is the published 3:1 linear:full ratio. `pos` then applies only
    # to the surviving global layers (GDN carries position in its recurrence), so
    # pos="nope" + hybrid_every=4 is the Kimi Linear config and retests the NoPE
    # conditional from docs/nope-verdict-audit.md.
    hybrid_every: int | None = None
    gdn_chunk: int = 64        # chunk-parallel block length (see gdn_chunkwise numerics)
    gdn_conv_kernel: int = 4   # short causal depthwise conv on q/k/v, as published
    gdn_fused: bool = True     # fused Triton kernel; False forces the reference path
    gdn_gate: str = "scalar"   # "scalar" = Gated DeltaNet | "channel" = KDA (per-channel)
    global_attn: str = "gqa"   # global-attention layer in a hybrid: "gqa" | "mla"
    mla_kv_lora: int = 512     # MLA latent width == cached values/token (NoPE: no rope dims)
    qk_norm: bool = False      # RMSNorm on q/k, head_dim variant (Qwen3/Gemma-3)
    fused_ce: bool = False     # fused linear+cross-entropy (Liger); 40-44% off train memory
    # optim / schedule
    optimizer: str = "adamw"  # "adamw" | "muon" (Muon on block matrices + AdamW on the rest)
    lr: float = 3e-4
    min_lr: float = 3e-5
    # Muon LR for the matrix params (only read when optimizer="muon"). Muon's orthogonalized
    # updates are on a different scale than AdamW's — the reference impl default is 0.02.
    # The schedule keeps one shape: muon groups get lr_schedule(step) * (muon_lr / lr).
    muon_lr: float = 0.02
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 100
    max_steps: int = 1000
    lr_decay_steps: int = 1000
    # data / io
    batch_size: int = 16
    grad_accum: int = 1
    # AUTHORITATIVE global batch, in tokens, when non-zero. grad_accum is then DERIVED per
    # world size (see microlab.train.distributed.batch_geometry) so the optimizer sees the
    # same batch on 1 GPU and on 8. Leave 0 to use grad_accum literally.
    tokens_per_step: int = 0
    eval_interval: int = 250
    eval_iters: int = 50
    ckpt_interval: int = 500
    ckpt_keep: int = 3  # 0 disables pruning; only the last N ckpt_*.pt files are kept
    # >0: checkpoints at multiples of this step are permanent (never pruned) — a research
    # trajectory kept alongside the rolling recovery window. Must divide ckpt_interval.
    ckpt_milestone_interval: int = 0
    grad_checkpoint: bool = False  # recompute activations backward: ~30x less act memory
    compile: bool = False          # torch.compile the model (CUDA; first step compiles)
    compile_mode: str = "default"  # "max-autotune" autotunes Triton kernels (slow compile,
    #                                ~29% faster steady-state on the 1B — worth it for long runs)
    log_interval: int = 50
    # Seconds a SINGLE training step may take before the process dumps every thread's
    # stack and dies. 0 disables it.
    #
    # A rented H100 froze at step 10 with 103 of 105 shards fetched, no error, and the
    # provider still reporting 100% GPU utilisation. It billed until a human noticed, and
    # left nothing to diagnose from — the run had to be reproduced to be understood, and
    # it did not reproduce locally. A hang that reports its own stack costs one dump; a
    # hang that stays silent costs the box, and then the next box.
    #
    # Deliberately fatal rather than a warning: the supervisor's resume path already
    # recovers from a dead trainer, so failing loudly re-provisions from the last
    # checkpoint, while hanging quietly bills at full rate forever.
    step_timeout_s: float = 0.0
    # Which shard dir this run trains on. None -> take it from --data-dir (the historical
    # behaviour). SET IT for any ablation whose intervention IS the data: the code
    # repetition lane's two arms were byte-identical configs distinguished only by a
    # --data-dir in a launcher script under /tmp, so the experiment was not legible from
    # the repo at all. scripts/preflight_lane.py hard-fails on identical arms for exactly
    # this reason.
    data_dir: str | None = None
    out_dir: str = "runs/pretrain"
    device: str = "cuda"
    dtype: str = "bfloat16"
    seed: int = 1337
