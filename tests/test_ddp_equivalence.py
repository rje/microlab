"""THE equivalence test: world_size 1 and world_size 2 must train the SAME model.

`tests/test_migration_safe_loader.py` proves the two world sizes consume the same DATA.
This proves they follow the same OPTIMIZER TRAJECTORY, which is a strictly stronger claim
and the one that makes a mid-run migration legitimate rather than merely plausible.

It is also the test that catches the Muon-under-DDP failure named in
docs/cloud-training-plan.md: Muon orthogonalises each matrix via Newton-Schulz, and
gradients must be all-reduced BEFORE that step. If any part of the update is computed from
a rank's LOCAL gradient, every rank orthogonalises a different matrix, the result is not
Muon at all — and the loss curve still looks entirely plausible. Nothing else we have would
notice.

Runs on gloo/CPU with 2 spawned processes, so it needs no second GPU and is part of the
ordinary suite rather than something only the cloud can check.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

BLOCK, TPS = 16, 128          # 8 sequences per step
STEPS = 6


def _write_shards(d: str) -> None:
    rng = np.random.default_rng(0)
    arr = rng.integers(1, 200, size=20_000, dtype=np.uint16)
    arr.tofile(os.path.join(d, "train-00000.bin"))
    arr.tofile(os.path.join(d, "val-00000.bin"))
    for split in ("train", "val"):
        with open(os.path.join(d, f"{split}-manifest.json"), "w") as f:
            json.dump({"split": split, "dtype": "uint16",
                       "shards": [{"file": f"{split}-00000.bin", "tokens": len(arr)}],
                       "total_tokens": len(arr)}, f)


def _free_port() -> int:
    """A port the OS says is free. A fixed port collides with the previous world size's
    store, which lingers briefly after teardown."""
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank: int, world: int, data_dir: str, out: str, optimizer: str,
            port: int) -> None:
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK="0",
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from microlab.data.shard_dataset import ShardDataset
    from microlab.train import distributed as dist_util
    from microlab.train.config import RunConfig
    from microlab.train.trainer import Trainer

    dist_util.setup("cpu", backend="gloo")
    cfg = RunConfig(
        vocab_size=256, block_size=BLOCK, n_layer=2, n_head=2, n_embd=32, dropout=0.0,
        norm="rms", pos="nope", mlp="swiglu", optimizer=optimizer, lr=1e-3, min_lr=1e-4,
        muon_lr=1e-2, warmup_steps=1, max_steps=STEPS, lr_decay_steps=STEPS,
        batch_size=1, tokens_per_step=TPS, compile=False, device="cpu",
        eval_interval=10**9, ckpt_interval=10**9, ckpt_milestone_interval=0,
        log_interval=10**9, out_dir=os.path.join(out, f"r{rank}"), seed=1234,
    )
    t = Trainer(cfg, ShardDataset(data_dir, "train"))
    losses = [t.train_step() for _ in range(STEPS)]
    if rank == 0:
        flat = torch.cat([p.detach().reshape(-1) for p in t.raw_model.parameters()])
        with open(os.path.join(out, f"result-ws{world}.json"), "w") as f:
            json.dump({"losses": losses,
                       "param_sum": float(flat.sum()),
                       "param_absmean": float(flat.abs().mean()),
                       "grad_accum": t.grad_accum,
                       "seqs_per_step": t.seqs_per_step}, f)
    dist_util.teardown()


def _run(world: int, data_dir: str, out: str, optimizer: str) -> dict:
    """ALWAYS spawn, including world_size 1.

    Running the single-rank case in-process leaves RANK/WORLD_SIZE in this interpreter's
    environment, and `dist_util.is_distributed()` reads exactly those — so every test that
    ran afterwards believed it was distributed. That polluted 23 unrelated tests in the
    full suite while passing in isolation, which is the most misleading shape a test bug
    can take.
    """
    port = _free_port()
    mp.spawn(_worker, args=(world, data_dir, out, optimizer, port),
             nprocs=world, join=True)
    with open(os.path.join(out, f"result-ws{world}.json")) as f:
        return json.load(f)


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_world_size_1_and_2_train_identically(optimizer):
    with tempfile.TemporaryDirectory() as d:
        data = os.path.join(d, "data")
        os.makedirs(data)
        _write_shards(data)
        a = _run(1, data, d, optimizer)
        b = _run(2, data, d, optimizer)

    # geometry: same sequences per step, different accumulation
    assert a["seqs_per_step"] == b["seqs_per_step"] == TPS // BLOCK
    assert a["grad_accum"] == 8 and b["grad_accum"] == 4

    for i, (la, lb) in enumerate(zip(a["losses"], b["losses"], strict=True)):
        assert la == pytest.approx(lb, rel=2e-4, abs=2e-5), (
            f"{optimizer}: loss diverged at step {i}: ws1={la:.8f} ws2={lb:.8f}. "
            f"The two world sizes are not training the same model.")

    assert a["param_sum"] == pytest.approx(b["param_sum"], rel=1e-4, abs=1e-4)
    assert a["param_absmean"] == pytest.approx(b["param_absmean"], rel=1e-4, abs=1e-6)


def test_muon_is_actually_being_exercised():
    """Guard the guard: if the muon parametrisation silently fell back to AdamW, the test
    above would pass while proving nothing about the failure it exists to catch."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from microlab.model.reference.variants import VariantConfig, VariantGPT
    from microlab.train.muon import build_muon_param_groups

    m = VariantGPT(VariantConfig(vocab_size=256, block_size=BLOCK, n_layer=2, n_head=2,
                                 n_embd=32))
    muon_params, adamw_params = build_muon_param_groups(m)
    # Muon must own the transformer BLOCK MATRICES. If that split degenerated to empty,
    # the equivalence test above would be exercising plain AdamW under a Muon label and
    # would prove nothing about the Newton-Schulz ordering it exists to check.
    assert muon_params, "no parameters routed to Muon — the equivalence test is vacuous"
    assert adamw_params, "no parameters routed to AdamW (embeddings/norms should be)"
    assert all(p.ndim >= 2 for p in muon_params), "Muon takes matrices, not 1-D params"
