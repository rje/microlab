"""The data stream must be invariant to world size, or a run cannot migrate.

The plan is to start the 1B locally and possibly finish it on 8xH100. That is only the
SAME run if step N sees the same sequences either side of the move. The legacy
`get_batch` cannot promise this: it picks shards from a `torch.Generator` whose state is
serialised into the checkpoint, so the stream depends on how many processes drew from it
and in what order.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from microlab.data.shard_dataset import ShardDataset

BLOCK = 16
SEED = 1337


@pytest.fixture()
def ds(tmp_path):
    rng = np.random.default_rng(0)
    shards = []
    for i in range(3):
        n = 4000 + 1000 * i
        arr = rng.integers(1, 500, size=n, dtype=np.uint16)
        name = f"train-{i:05d}.bin"
        arr.tofile(tmp_path / name)
        shards.append({"file": name, "tokens": n})
    (tmp_path / "train-manifest.json").write_text(json.dumps(
        {"split": "train", "dtype": "uint16", "shards": shards,
         "total_tokens": sum(s["tokens"] for s in shards)}))
    return ShardDataset(str(tmp_path), "train")


def _global_batch(ds, world_size, grad_accum, batch_size, step, seqs_per_step):
    """Every sequence the whole job consumes at `step`, in index order."""
    seen = []
    for micro in range(grad_accum):
        for rank in range(world_size):
            seen += ds.global_indices(step, micro, rank, world_size,
                                      batch_size, seqs_per_step)
    return sorted(seen)


def test_global_batch_is_identical_across_world_sizes(ds):
    """THE migration guarantee: 1 GPU x accum 16 == 8 GPUs x accum 2."""
    seqs = 16
    local = _global_batch(ds, world_size=1, grad_accum=16, batch_size=1,
                          step=7, seqs_per_step=seqs)
    cloud = _global_batch(ds, world_size=8, grad_accum=2, batch_size=1,
                          step=7, seqs_per_step=seqs)
    assert local == cloud == list(range(7 * seqs, 8 * seqs))


@pytest.mark.parametrize("ws,accum", [(1, 16), (2, 8), (4, 4), (8, 2), (16, 1)])
def test_every_valid_split_covers_each_sequence_exactly_once(ds, ws, accum):
    got = _global_batch(ds, ws, accum, batch_size=1, step=3, seqs_per_step=16)
    assert got == list(range(3 * 16, 4 * 16)), f"world_size={ws} accum={accum}"


def test_sequence_content_depends_only_on_index_and_seed(ds):
    """Same index -> same tokens, no matter who asks or when."""
    a, _ = ds.sequence_at(42, BLOCK, SEED)
    b, _ = ds.sequence_at(42, BLOCK, SEED)
    assert a.tolist() == b.tolist()


def test_different_indices_give_different_windows(ds):
    """A weak hash would map adjacent indices to overlapping offsets."""
    seqs = [tuple(ds.sequence_at(i, BLOCK, SEED)[0].tolist()) for i in range(40)]
    assert len(set(seqs)) >= 38, "adjacent sequences collide — hash is not mixing"


def test_seed_changes_the_stream(ds):
    a, _ = ds.sequence_at(5, BLOCK, 1337)
    b, _ = ds.sequence_at(5, BLOCK, 1338)
    assert a.tolist() != b.tolist()


def test_targets_are_inputs_shifted_by_one(ds):
    x, y = ds.sequence_at(11, BLOCK, SEED)
    assert x[1:].tolist() == y[:-1].tolist()


def test_indivisible_layout_raises(ds):
    """tokens_per_step must divide by world_size*batch_size*block_size, or the global
    batch silently changes size on migration."""
    with pytest.raises(ValueError, match="exceeds seqs_per_step"):
        ds.global_indices(step=0, micro_step=9, rank=0, world_size=8,
                          batch_size=1, seqs_per_step=16)


def test_resume_needs_only_the_step_number(ds):
    """No generator state: step 100 is reproducible from scratch."""
    fresh = _global_batch(ds, 1, 16, 1, step=100, seqs_per_step=16)
    assert fresh == list(range(1600, 1616))
    x1, _ = ds.sequence_at(1600, BLOCK, SEED)
    x2, _ = ds.sequence_at(1600, BLOCK, SEED)
    assert x1.tolist() == x2.tolist()


def test_batch_size_above_one_packs_within_a_rank(ds):
    idx = ds.global_indices(step=0, micro_step=0, rank=0, world_size=2,
                            batch_size=2, seqs_per_step=8)
    assert idx == [0, 1]
    idx_r1 = ds.global_indices(step=0, micro_step=0, rank=1, world_size=2,
                               batch_size=2, seqs_per_step=8)
    assert idx_r1 == [2, 3]
    assert set(idx).isdisjoint(idx_r1)


def test_get_batch_indexed_shapes(ds):
    x, y = ds.get_batch_indexed(BLOCK, [0, 1, 2], SEED)
    assert x.shape == (3, BLOCK) and y.shape == (3, BLOCK)
