import numpy as np
import torch

from microlab.data.prepare import strip_contamination, write_shards
from microlab.data.shard_dataset import ShardDataset


def test_strip_contamination():
    texts = ["clean", "has EVAL_Q inside", "fine"]
    assert list(strip_contamination(texts, ["EVAL_Q"])) == ["clean", "fine"]


def test_write_shards_and_readback(tmp_path):
    toks = list(range(1000))
    man = write_shards(iter(toks), str(tmp_path), split="train", shard_size=300)
    assert man["total_tokens"] == 1000 and len(man["shards"]) == 4  # 300,300,300,100
    read = []
    for s in man["shards"]:
        read.extend(np.fromfile(tmp_path / s["file"], dtype=np.uint16).tolist())
    assert read == toks


def test_shard_dataset_batch_shapes_and_shift(tmp_path):
    write_shards(iter(range(5000)), str(tmp_path), split="train", shard_size=5000)
    ds = ShardDataset(str(tmp_path), split="train")
    g = torch.Generator().manual_seed(0)
    x, y = ds.get_batch(block_size=16, batch_size=8, generator=g)
    assert x.shape == (8, 16) and y.shape == (8, 16)
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_shard_dataset_deterministic(tmp_path):
    write_shards(iter(range(5000)), str(tmp_path), split="train", shard_size=5000)
    ds = ShardDataset(str(tmp_path), split="train")
    x1, _ = ds.get_batch(16, 4, generator=torch.Generator().manual_seed(1))
    x2, _ = ds.get_batch(16, 4, generator=torch.Generator().manual_seed(1))
    assert torch.equal(x1, x2)
