import pytest
import torch

from microlab.train.config import RunConfig
from microlab.train.trainer import TensorData, Trainer, get_lr


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
