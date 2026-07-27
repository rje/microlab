"""configs/1b-4k.py: stage-1 context extension of the ORIGINAL MHA 1B — same geometry as
configs/1b.py with block_size 4x'd to 4096 and rope_base raised 10k -> 100k (ABF), Muon,
~0.5B tokens of gentle continued training into runs/1b-4k."""

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _load(path):
    spec = importlib.util.spec_from_file_location("cfg_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.config


def test_1b_4k_config_values():
    cfg = _load(_ROOT / "configs" / "1b-4k.py")
    base = _load(_ROOT / "configs" / "1b.py")
    # identical geometry to the base 1B except the context length and RoPE base
    for f in ("vocab_size", "n_layer", "n_head", "n_embd", "norm", "pos", "mlp"):
        assert getattr(cfg, f) == getattr(base, f), f
    assert cfg.block_size == 4096
    assert cfg.rope_base == 100000.0  # ABF: adjusted base frequency, 10x for a 4x window
    assert cfg.n_kv_head is None  # deliberately the ORIGINAL MHA model, not the GQA variant
    assert cfg.optimizer == "muon"
    assert cfg.out_dir == "runs/1b-4k"
    # ~0.5B tokens of continued training
    tokens = cfg.batch_size * cfg.grad_accum * cfg.block_size * cfg.max_steps
    assert 0.4e9 <= tokens <= 0.65e9
    # effective tokens/step stays within 2x of the 8*64*1024=524288 convention
    toks_step = cfg.batch_size * cfg.grad_accum * cfg.block_size
    assert 524288 / 2 <= toks_step <= 524288 * 2
    assert cfg.max_steps == cfg.lr_decay_steps  # one full anneal, ends settled
    # adaptation of a trained model, not fresh pretraining: gentle LRs (1e-4 -> 1e-5)
    assert cfg.muon_lr < 0.02
    assert cfg.lr <= 1e-4 and cfg.min_lr <= 1e-5
    assert cfg.warmup_steps <= cfg.max_steps // 10
    # curve resolution + two-tier checkpointing (mirrors the 1b-gqa uptrain conventions)
    assert cfg.eval_interval == 100
    assert cfg.ckpt_interval > 0 and cfg.ckpt_milestone_interval % cfg.ckpt_interval == 0
    assert cfg.compile_mode == "max-autotune-no-cudagraphs"  # tied-weights landmine
    # a fresh data-sampling stream: don't replay the base run's exact batch sequence
    assert cfg.seed != base.seed
