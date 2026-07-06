import pytest
import torch

from microlab.train.config import RunConfig
from microlab.train.trainer import TensorData, Trainer, get_lr, gpu_scalars


def _cfg(**kw):
    base = dict(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32,
                warmup_steps=5, max_steps=20, lr_decay_steps=20, batch_size=8,
                eval_interval=1000, ckpt_interval=1000, device="cpu", dtype="float32")
    base.update(kw)
    return RunConfig(**base)


def test_lr_schedule_shape():
    c = _cfg(lr=1.0, min_lr=0.1, warmup_steps=10, lr_decay_steps=110)
    assert get_lr(0, c) == pytest.approx(0.0, abs=1e-6)      # warmup starts at 0
    assert get_lr(10, c) == pytest.approx(1.0, abs=1e-6)     # peak at end of warmup
    assert get_lr(110, c) == pytest.approx(0.1, abs=1e-6)    # decays to min
    assert get_lr(200, c) == pytest.approx(0.1, abs=1e-6)    # stays at min
    assert 0.1 < get_lr(60, c) < 1.0                         # mid-decay in between


def test_training_reduces_loss_cpu():
    torch.manual_seed(0)
    data = TensorData(torch.randint(0, 64, (4000,)))
    tr = Trainer(_cfg(), data, data)
    stats = tr.train()
    assert stats["history"][-1] < stats["history"][0]


def test_resume_equivalence_cpu(tmp_path):
    # An uninterrupted 20-step run must match a 10-step + resume + 10-step run.
    data = TensorData(torch.randint(0, 64, (4000,), generator=torch.Generator().manual_seed(42)))
    a = Trainer(_cfg(max_steps=20), data, None)
    a.train()
    ck = str(tmp_path / "ck.pt")
    b = Trainer(_cfg(max_steps=10), data, None)
    b.train()
    b.save_checkpoint(ck)
    c = Trainer(_cfg(max_steps=20), data, None)
    c.load_checkpoint(ck)
    c.train()  # resumes from step 10 to 20
    # same model params after resume as the uninterrupted run
    for pa, pc in zip(a.model.parameters(), c.model.parameters(), strict=True):
        assert torch.allclose(pa, pc, atol=1e-5)


def test_train_returns_stats_keys_with_tokenizer_none(tmp_path):
    # TB logging is side-effect only: tokenizer=None must keep the stats contract intact.
    torch.manual_seed(0)
    data = TensorData(torch.randint(0, 64, (4000,)))
    tr = Trainer(_cfg(out_dir=str(tmp_path), max_steps=3), data, data, tokenizer=None)
    stats = tr.train()
    assert set(stats) == {"final_loss", "history", "val_loss", "step"}
    assert stats["step"] == 3


def test_checkpoint_pruning(tmp_path):
    # Only the last `ckpt_keep` checkpoints should survive on disk.
    torch.manual_seed(0)
    data = TensorData(torch.randint(0, 64, (4000,)))
    tr = Trainer(
        _cfg(out_dir=str(tmp_path), max_steps=6, ckpt_interval=2, ckpt_keep=2), data, data
    )
    tr.train()
    ckpts = sorted(tmp_path.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    assert [p.stem for p in ckpts] == ["ckpt_4", "ckpt_6"]


def test_checkpoint_milestone_retention(tmp_path):
    # Two-tier retention: milestones (multiples of ckpt_milestone_interval) are permanent,
    # AND the last `ckpt_keep` rolling checkpoints survive for crash recovery. A milestone
    # outside the rolling window must NOT be pruned.
    torch.manual_seed(0)
    data = TensorData(torch.randint(0, 64, (4000,)))
    tr = Trainer(
        _cfg(out_dir=str(tmp_path), max_steps=12, ckpt_interval=2, ckpt_keep=2,
             ckpt_milestone_interval=6),
        data, data,
    )
    tr.train()
    ckpts = sorted(tmp_path.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    # milestone 6 survives though it's outside the last-2 rolling window (10, 12);
    # 12 is both a milestone and the latest rolling checkpoint.
    assert [p.stem for p in ckpts] == ["ckpt_6", "ckpt_10", "ckpt_12"]


def test_milestone_interval_must_divide_ckpt_interval(tmp_path):
    # A milestone cadence not divisible by ckpt_interval means milestone steps are never
    # actually checkpointed — a silent no-op. Fail loudly at construction instead.
    data = TensorData(torch.randint(0, 64, (4000,)))
    with pytest.raises(ValueError, match="multiple of ckpt_interval"):
        Trainer(
            _cfg(out_dir=str(tmp_path), ckpt_interval=200, ckpt_milestone_interval=300),
            data, data,
        )


def test_gpu_scalars_empty_off_cuda():
    # Off CUDA there is nothing to report — no keys, regardless of the NVML flag. This is the
    # path the CPU test suite and CPU training runs take; it must never touch torch.cuda.*.
    assert gpu_scalars("cpu", include_nvml=True) == {}
    assert gpu_scalars("cpu", include_nvml=False) == {}


def test_cpu_trainer_disables_gpu_telemetry(tmp_path):
    # A CPU Trainer must not claim NVML telemetry, so its logging stays memory-free and never
    # calls the NVML helpers (which would raise off-CUDA).
    data = TensorData(torch.randint(0, 64, (4000,)))
    tr = Trainer(_cfg(out_dir=str(tmp_path), max_steps=1), data, data)
    assert tr._gpu_nvml is False


def test_tensorboard_event_file_written(tmp_path):
    # With tensorboard installed, a tiny run must emit an event file into out_dir
    # and must not crash the training loop.
    torch.manual_seed(0)
    data = TensorData(torch.randint(0, 64, (4000,)))
    tr = Trainer(
        _cfg(out_dir=str(tmp_path), max_steps=3, log_interval=1, eval_interval=2),
        data,
        data,
    )
    tr.train()
    events = list(tmp_path.glob("events.out.tfevents.*"))
    assert events, "expected a TensorBoard event file in out_dir"


def test_grad_checkpoint_and_compile_flags(tmp_path):
    # Both flags must run a short training and still checkpoint/resume via the RAW model
    # (torch.compile prefixes state_dict keys with _orig_mod. if you save the wrapper).
    torch.manual_seed(0)
    data = TensorData(torch.randint(0, 64, (4000,)))
    tr = Trainer(_cfg(out_dir=str(tmp_path), max_steps=3, grad_checkpoint=True), data, data)
    stats = tr.train()
    assert stats["step"] == 3
    ck = str(tmp_path / "ck.pt")
    tr.save_checkpoint(ck)
    tr2 = Trainer(_cfg(max_steps=3), data, data)
    tr2.load_checkpoint(ck)  # keys must match the uncompiled/unwrapped model


@pytest.mark.gpu
def test_compile_flag_on_cuda(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    data = TensorData(torch.randint(0, 64, (8000,)))
    tr = Trainer(_cfg(device="cuda", dtype="bfloat16", out_dir=str(tmp_path), max_steps=3,
                      compile=True), data, data)
    assert tr.train()["step"] == 3


@pytest.mark.gpu
def test_trainer_on_cuda_bf16():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    data = TensorData(torch.randint(0, 64, (8000,)))
    tr = Trainer(_cfg(device="cuda", dtype="bfloat16", n_embd=64, max_steps=40), data, data)
    stats = tr.train()
    assert stats["history"][-1] < stats["history"][0]


@pytest.mark.gpu
def test_checkpoint_resume_on_cuda(tmp_path):
    # Regression: torch.load(map_location="cuda") moves the RNG ByteTensors onto CUDA,
    # but set_rng_state_all needs CPU bytes — resume must handle that.
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    data = TensorData(torch.randint(0, 64, (5000,)))
    tr = Trainer(_cfg(device="cuda", dtype="bfloat16", n_embd=64, max_steps=20), data, data)
    tr.train()
    ck = str(tmp_path / "ck.pt")
    tr.save_checkpoint(ck)
    tr2 = Trainer(_cfg(device="cuda", dtype="bfloat16", n_embd=64, max_steps=20), data, data)
    tr2.load_checkpoint(ck)  # must not raise
    assert tr2.step == 20
