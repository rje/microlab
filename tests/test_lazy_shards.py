"""Lazy shard fetch: start training before the corpus has finished downloading.

On preemptible hardware the corpus pull is paid on EVERY re-provision — 9 minutes from
US-West, 54 from Asia, measured. Training reads shards in random order and touches one per
sequence, so requiring all 105 up front is a self-imposed cost. These tests pin the
properties that make deferring the download safe:

  * the dataset knows its own size before ANY shard exists, so sequence->shard mapping is
    identical local or remote;
  * a shard is fetched at most once;
  * a truncated fetch RAISES rather than silently training on short data;
  * a partially-downloaded file is never visible as a complete one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from microlab.data.shard_dataset import ShardDataset

BLOCK, SEED = 32, 1337


def _manifest(d: Path, n_shards=4, tokens=5000):
    shards = []
    for i in range(n_shards):
        shards.append({"file": f"train-{i:05d}.bin", "tokens": tokens})
    (d / "train-manifest.json").write_text(json.dumps(
        {"split": "train", "dtype": "uint16", "shards": shards,
         "total_tokens": n_shards * tokens}))
    return shards


def _write(d: Path, name: str, tokens: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    rng.integers(1, 500, size=tokens, dtype=np.uint16).tofile(d / name)


def test_dataset_knows_its_size_with_no_shards_on_disk(tmp_path):
    """The manifest is the source of truth, so index->shard mapping is fixed before any
    download happens. Without this, a lazily-fetched run would map sequences differently
    from an eager one — a silently different experiment."""
    _manifest(tmp_path)
    ds = ShardDataset(str(tmp_path), "train", fetch=lambda name, dest: None)
    assert ds.total_tokens == 20_000
    assert ds.lengths == [5000] * 4
    assert not list(tmp_path.glob("*.bin"))


def test_fetches_only_the_shard_it_needs(tmp_path):
    _manifest(tmp_path)
    src = tmp_path / "remote"
    src.mkdir()
    for i in range(4):
        _write(src, f"train-{i:05d}.bin", 5000, seed=i)
    fetched = []

    def fetch(name, dest):
        fetched.append(name)
        dest.write_bytes((src / name).read_bytes())

    ds = ShardDataset(str(tmp_path), "train", fetch=fetch)
    ds.sequence_at(0, BLOCK, SEED)
    assert len(fetched) == 1, f"one sequence should need one shard, fetched {fetched}"


def test_a_shard_is_fetched_at_most_once(tmp_path):
    _manifest(tmp_path, n_shards=1)
    src = tmp_path / "remote"
    src.mkdir()
    _write(src, "train-00000.bin", 5000)
    calls = []

    def fetch(name, dest):
        calls.append(name)
        dest.write_bytes((src / name).read_bytes())

    ds = ShardDataset(str(tmp_path), "train", fetch=fetch)
    for i in range(25):
        ds.sequence_at(i, BLOCK, SEED)
    assert len(calls) == 1, f"cached after first fetch, got {len(calls)} fetches"


def test_lazy_and_eager_produce_identical_sequences(tmp_path):
    """THE correctness property: deferring the download must not change training."""
    eager = tmp_path / "eager"
    lazy = tmp_path / "lazy"
    for d in (eager, lazy):
        d.mkdir()
        _manifest(d)
    for i in range(4):
        _write(eager, f"train-{i:05d}.bin", 5000, seed=i)

    def fetch(name, dest):
        dest.write_bytes((eager / name).read_bytes())

    a = ShardDataset(str(eager), "train")
    b = ShardDataset(str(lazy), "train", fetch=fetch)
    for i in (0, 1, 7, 30, 99):
        xa, ya = a.sequence_at(i, BLOCK, SEED)
        xb, yb = b.sequence_at(i, BLOCK, SEED)
        assert xa.tolist() == xb.tolist(), f"sequence {i} differs"
        assert ya.tolist() == yb.tolist()


def test_truncated_fetch_raises(tmp_path):
    """A short shard is still valid uint16 and would train on truncated data silently."""
    _manifest(tmp_path, n_shards=1)

    def fetch(name, dest):
        np.zeros(100, dtype=np.uint16).tofile(dest)      # manifest says 5000

    ds = ShardDataset(str(tmp_path), "train", fetch=fetch)
    with pytest.raises(ValueError, match="manifest implies"):
        ds.sequence_at(0, BLOCK, SEED)


def test_partial_download_is_never_visible_as_complete(tmp_path):
    """Fetch writes to .part and renames, so a crash mid-download cannot leave a file that
    a later run treats as a finished shard."""
    _manifest(tmp_path, n_shards=1)
    seen = {}

    def fetch(name, dest):
        seen["dest"] = Path(dest)
        np.zeros(5000, dtype=np.uint16).tofile(dest)

    ds = ShardDataset(str(tmp_path), "train", fetch=fetch)
    ds.sequence_at(0, BLOCK, SEED)
    assert ".part" in seen["dest"].name, "download must land on a temp name"
    assert (tmp_path / "train-00000.bin").exists()
    assert not list(tmp_path.glob("*.part*")), "temp file should be renamed away"


def test_eager_mode_still_validates_the_manifest(tmp_path):
    """A manifest disagreeing with the files would shift every sequence index."""
    _manifest(tmp_path, n_shards=1, tokens=5000)
    _write(tmp_path, "train-00000.bin", 4000)            # actually 4000
    with pytest.raises(ValueError, match="manifest says"):
        ShardDataset(str(tmp_path), "train")


def test_arrays_property_still_works_for_eager_callers(tmp_path):
    _manifest(tmp_path, n_shards=2)
    for i in range(2):
        _write(tmp_path, f"train-{i:05d}.bin", 5000, seed=i)
    ds = ShardDataset(str(tmp_path), "train")
    assert len(ds.arrays) == 2
    assert all(len(a) == 5000 for a in ds.arrays)


def test_concurrent_fetch_of_one_shard_downloads_it_once(tmp_path):
    """Main thread and prefetch threads must not race on the same shard.

    Both wrote the same `.part` file: boto3 writes multipart chunks at offsets, so two
    writers corrupt it, and whichever renames first leaves the other stat-ing a path that
    no longer exists. A wedged run on rented hardware — GPU at 0% with every shard already
    cached — is what sent me looking for this.
    """
    import threading
    import time as _t

    _manifest(tmp_path, n_shards=1)
    src = tmp_path / "remote"
    src.mkdir()
    _write(src, "train-00000.bin", 5000)
    calls = []

    def slow_fetch(name, dest):
        calls.append(name)
        _t.sleep(0.15)                      # widen the race window
        dest.write_bytes((src / name).read_bytes())

    ds = ShardDataset(str(tmp_path), "train", fetch=slow_fetch)
    errors = []

    def worker():
        try:
            ds.sequence_at(0, BLOCK, SEED)
        except Exception as e:              # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)

    assert not errors, f"concurrent access raised: {errors}"
    assert len(calls) == 1, f"shard fetched {len(calls)} times, expected exactly 1"
    assert not any(t.is_alive() for t in threads), "a thread hung"


def test_distinct_shards_use_distinct_temp_files(tmp_path):
    """A shared temp name would have concurrent downloads overwrite each other."""
    _manifest(tmp_path, n_shards=3)
    seen = []

    def fetch(name, dest):
        seen.append(Path(dest).name)
        _write(Path(dest).parent, Path(dest).name, 5000)

    ds = ShardDataset(str(tmp_path), "train", fetch=fetch)
    for i in range(3):
        ds._array(i)
    assert len(set(seen)) == 3, f"temp names collided: {seen}"
