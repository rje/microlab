"""configs/1b-gqa.py: the GQA uptrain must target the converted 1B geometry (14 query
heads, 2 KV heads), use Muon, run ~1B tokens (~5% of the base run's 21B — the GQA
paper's uptraining proportion), and keep the checkpoint cadence internally consistent."""

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _load(path):
    spec = importlib.util.spec_from_file_location("cfg_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.config


def test_1b_gqa_config_values():
    cfg = _load(_ROOT / "configs" / "1b-gqa.py")
    base = _load(_ROOT / "configs" / "1b.py")
    # identical geometry to the base run except attention grouping
    for f in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd", "norm", "pos", "mlp"):
        assert getattr(cfg, f) == getattr(base, f), f
    assert cfg.n_kv_head == 2
    assert base.n_head % cfg.n_kv_head == 0
    assert cfg.optimizer == "muon"
    assert cfg.out_dir == "runs/1b-gqa"
    # ~1B tokens: the paper's alpha=0.05 of the 21B-token base recipe
    tokens = cfg.batch_size * cfg.grad_accum * cfg.block_size * cfg.max_steps
    assert 0.9e9 <= tokens <= 1.2e9
    assert cfg.max_steps == cfg.lr_decay_steps
    # recovery, not fresh pretraining: gentler than the fresh-run Muon peak (0.02)
    assert cfg.muon_lr < 0.02
    assert cfg.lr < base.lr
    # recovery-curve resolution + sane two-tier checkpointing for a 2000-step run
    assert cfg.eval_interval == 100
    assert cfg.ckpt_interval == 100
    assert cfg.ckpt_milestone_interval % cfg.ckpt_interval == 0
    assert cfg.compile_mode == "max-autotune-no-cudagraphs"  # tied-weights landmine
    # a fresh data-sampling stream: don't replay the base run's exact batch sequence
    assert cfg.seed != base.seed
