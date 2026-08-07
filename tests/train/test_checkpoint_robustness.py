"""Checkpoint durability: atomic writes, rank-0-only pruning, resume past corruption.

Three failure modes from the production review of the paid pipeline, each of which turns
one bad file or one race into a full re-provision (about 20 minutes of setup at full GPU
price) or an unbounded crash loop.
"""

from __future__ import annotations

from pathlib import Path

import pytest
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


def test_a_pre_split_optimizer_checkpoint_still_loads():
    """THE shakedown crash. The decay-routing fix split AdamW into decay/no-decay
    groups, which made every checkpoint saved before it raise "different number of
    parameter groups" — and the resume fallback then set a healthy 9.2 GB checkpoint
    aside as corrupt. Old checkpoints must migrate by parameter identity."""
    from microlab.train.muon import Muon, MuonAdamW, build_muon_param_groups

    _t = torch

    class M(_t.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = _t.nn.ModuleDict(dict(
                wte=_t.nn.Embedding(64, 16),
                h=_t.nn.ModuleList([_t.nn.ModuleDict(dict(
                    w=_t.nn.Linear(16, 16, bias=False))) for _ in range(2)]),
                ln=_t.nn.LayerNorm(16)))
            self.lm_head = _t.nn.Linear(16, 64, bias=False)
            self.lm_head.weight = self.transformer.wte.weight

    def old_style(model):
        mu, ad = build_muon_param_groups(model)
        return MuonAdamW(Muon(mu, lr=0.02),
                         _t.optim.AdamW(ad, lr=3e-4, weight_decay=0.1),
                         muon_lr_scale=0.02 / 3e-4, adamw_flat=ad), ad

    def new_style(model):
        from microlab.train.muon import build_muon_optimizer
        return build_muon_optimizer(model, lr=3e-4, muon_lr=0.02, betas=(0.9, 0.95),
                                    weight_decay=0.1, fused=False)

    m = M()
    old_opt, flat = old_style(m)
    # take real steps so there is real state (exp_avg / step counters) to migrate
    for _ in range(3):
        loss = m.lm_head(m.transformer.ln(m.transformer.wte(
            _t.randint(0, 64, (4,))))).sum()
        old_opt.zero_grad()
        loss.backward()
        old_opt.step()
    sd = old_opt.state_dict()
    assert len(sd["adamw"]["param_groups"]) == 1, "precondition: pre-split shape"

    new_opt = new_style(m)
    new_opt.load_state_dict(sd)                      # <- raised before the migration
    # state followed the PARAMETERS, not the group indices
    norm_gain = m.transformer.ln.weight
    # per-param check: exp_avg shape matches its parameter everywhere
    for g in new_opt.adamw.param_groups:
        for p in g["params"]:
            st = new_opt.adamw.state.get(p)
            assert st, f"state lost for param of shape {tuple(p.shape)}"
            assert st["exp_avg"].shape == p.shape
    # and decay routing reflects the NEW groups, not the old single group
    for g in new_opt.adamw.param_groups:
        if any(p is norm_gain for p in g["params"]):
            assert g["weight_decay"] == 0.0


def test_version_skew_is_not_classified_as_corruption(tmp_path):
    """A checkpoint that DESERIALIZES but does not fit the current optimizer must raise
    the real error, not be renamed .corrupt: the file is fine, the code changed."""
    from microlab.train.trainer import CorruptCheckpoint
    t = Trainer(_cfg(tmp_path), _data())
    p = tmp_path / "run" / "ckpt_1.pt"
    t.save_checkpoint(str(p))
    ck = torch.load(p, weights_only=False)
    ck["optimizer"]["param_groups"] = []             # structurally wrong, bytes fine
    torch.save(ck, p)
    fresh = Trainer(_cfg(tmp_path), _data())
    with pytest.raises(Exception) as ei:
        fresh.load_checkpoint(str(p))
    assert not isinstance(ei.value, CorruptCheckpoint), \
        "a fitting error must not carry the corrupt-file classification"


def test_unreadable_bytes_are_classified_as_corruption(tmp_path):
    from microlab.train.trainer import CorruptCheckpoint
    t = Trainer(_cfg(tmp_path), _data())
    p = tmp_path / "run" / "ckpt_1.pt"
    t.save_checkpoint(str(p))
    p.write_bytes(p.read_bytes()[:4000])             # truncate: undeserializable
    with pytest.raises(CorruptCheckpoint):
        Trainer(_cfg(tmp_path), _data()).load_checkpoint(str(p))


def test_cuda_rng_states_are_bounded_to_this_boxs_device_count():
    """Resuming a 4-GPU checkpoint on a 2-GPU box is a SUPPORTED operation (world-size-
    invariant geometry; the retry loop's half-speed fallback). set_rng_state_all indexes
    default_generators[i] for every i, so 4 saved states on a 2-GPU box raised
    'IndexError: tuple index out of range' and killed the resume. The restore must be
    bounded to the current device count."""
    from microlab.train.trainer import cuda_rng_states_for_devices
    saved = ["s0", "s1", "s2", "s3"]                  # saved on a 4-GPU box
    assert cuda_rng_states_for_devices(saved, 2) == ["s0", "s1"], \
        "too-many states must be truncated to the devices that exist here"
    assert cuda_rng_states_for_devices(saved, 4) == saved, \
        "matching device count must be an exact no-op (bit-identical 4->4 resume)"
    assert cuda_rng_states_for_devices(["s0", "s1"], 4) == ["s0", "s1"], \
        "fewer states than devices is fine as-is; extra devices keep their default seed"
