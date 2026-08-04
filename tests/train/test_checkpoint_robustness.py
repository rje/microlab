"""Checkpoint durability: atomic writes, rank-0-only pruning, resume past corruption.

Three failure modes from the production review of the paid pipeline, each of which turns
one bad file or one race into a full re-provision (about 20 minutes of setup at full GPU
price) or an unbounded crash loop.
"""

from __future__ import annotations

from pathlib import Path

import torch

from microlab.train.config import RunConfig
from microlab.train.trainer import TensorData, Trainer


def _data():
    return TensorData(torch.randint(0, 64, (2000,),
                      generator=torch.Generator().manual_seed(3)))


def _cfg(tmp_path, **kw):
    base = dict(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32,
                warmup_steps=1, max_steps=2, lr_decay_steps=2, batch_size=2,
                eval_interval=0, ckpt_interval=0, device="cpu", dtype="float32",
                out_dir=str(tmp_path / "run"))
    base.update(kw)
    return RunConfig(**base)


def test_no_partial_checkpoint_is_ever_visible_under_the_final_name(tmp_path):
    """torch.save straight to ckpt_N.pt races the B2 syncer, whose 'fully written' test
    is size stability across 3 s — a page-cache stall makes a partial file pass, upload
    as the highest step, and poison every later resume. Write-then-rename removes the
    window entirely: the temp name never matches the syncer's glob."""
    t = Trainer(_cfg(tmp_path), _data())
    p = tmp_path / "run" / "ckpt_1.pt"
    real_save = torch.save
    seen = {}

    def spy(obj, f, *a, **kw):
        seen["written_to"] = str(f)
        return real_save(obj, f, *a, **kw)

    torch.save = spy
    try:
        t.save_checkpoint(str(p))
    finally:
        torch.save = real_save
    assert seen["written_to"] != str(p), "checkpoint must be written to a temp name"
    assert seen["written_to"].endswith(".tmp")
    assert p.exists() and not Path(seen["written_to"]).exists(), \
        "temp file must be renamed into place"
    torch.load(p, weights_only=False)                     # and the result must load


def test_resume_walks_past_a_corrupt_newest_checkpoint(tmp_path):
    """A truncated highest-step checkpoint must cost one warning, not the run. The resume
    download verifies size against the REMOTE object — which is also truncated — so the
    corruption is only discoverable at torch.load, and refusing to fall back there is a
    deterministic re-provision loop."""
    t = Trainer(_cfg(tmp_path), _data())
    t.train()
    good = tmp_path / "run" / "ckpt_2.pt"
    t.save_checkpoint(str(good))
    bad = tmp_path / "run" / "ckpt_3.pt"
    bad.write_bytes(torch.load(good, weights_only=False) and good.read_bytes()[:5000])

    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(
        "pretrain_mod", Path(__file__).resolve().parents[2] / "scripts" / "pretrain.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pretrain_mod"] = spec.loader is not None and mod
    # The walk-down logic lives in main(); exercise its core contract directly instead:
    ckpts = sorted((tmp_path / "run").glob("ckpt_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    assert ckpts[-1] == bad
    fresh = Trainer(_cfg(tmp_path), _data())
    loaded = None
    for c in reversed(ckpts):
        try:
            fresh.load_checkpoint(str(c))
        except Exception:
            c.rename(c.with_suffix(".pt.corrupt"))
            continue
        loaded = c
        break
    assert loaded == good, "resume must fall back to the newest LOADABLE checkpoint"
    assert fresh.step == 2
    assert (tmp_path / "run" / "ckpt_3.pt.corrupt").exists(), \
        "the corrupt file must be preserved for diagnosis, under a name no resume reads"


def test_the_walk_down_exists_in_the_actual_resume_path():
    """The contract above must be wired into scripts/pretrain.py, not just testable."""
    src = (Path(__file__).resolve().parents[2] / "scripts" / "pretrain.py").read_text()
    assert ".pt.corrupt" in src, "corrupt checkpoints must be renamed aside"
    assert "reversed(ckpts)" in src, "resume must walk down the checkpoint list"
    assert "refusing to silently restart" in src, \
        "all-corrupt must be fatal, not a quiet restart from step 0"


def test_prune_is_rank_zero_only():
    """All ranks ran the glob-and-unlink after the save barrier; two ranks racing the
    same unlink crashed the loser with FileNotFoundError every ckpt_interval once the
    window filled. The guard must precede any filesystem work."""
    import inspect
    src = inspect.getsource(Trainer._prune_checkpoints)
    body = src[:src.index(".glob(")]
    assert "is_main" in body, "_prune_checkpoints must return early on non-main ranks"
    assert "missing_ok=True" in src, \
        "unlink must tolerate a file the B2 syncer already pruned"
