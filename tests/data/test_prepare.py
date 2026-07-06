import numpy as np
import torch

from microlab.data.prepare import (
    batched_token_chunks,
    parallel_token_chunks,
    strip_contamination,
    take_tokens,
    write_shards,
)
from microlab.data.shard_dataset import ShardDataset
from microlab.tokenizer.fast import FastTokenizer


def test_strip_contamination():
    texts = ["clean", "has EVAL_Q inside", "fine"]
    assert list(strip_contamination(texts, ["EVAL_Q"])) == ["clean", "fine"]


def test_write_shards_accepts_token_arrays(tmp_path):
    # the scale path: arrays in, exact shard_size shards out, bytes round-trip
    chunks = [np.arange(0, 250, dtype=np.uint16), np.arange(250, 1000, dtype=np.uint16)]
    man = write_shards(chunks, str(tmp_path), split="train", shard_size=300)
    assert man["total_tokens"] == 1000 and len(man["shards"]) == 4  # 300,300,300,100
    read = []
    for s in man["shards"]:
        read.extend(np.fromfile(tmp_path / s["file"], dtype=np.uint16).tolist())
    assert read == list(range(1000))


def test_batched_token_chunks_matches_per_doc(tmp_path):
    tok = FastTokenizer.train(["hello world", "the cat sat on mat", "a b c d e"] * 4,
                              vocab_size=300, save_path=str(tmp_path / "tok.json"))
    docs = ["hello world foo bar", "the cat sat", "a b c d e f g"]
    eot = tok.eot_token
    ref: list[int] = []
    for d in docs:
        ref.extend(tok.encode(d))
        ref.append(eot)
    got = np.concatenate(list(batched_token_chunks(tok, docs, eot, batch_docs=2)))
    assert got.tolist() == ref  # batched encoding == per-doc encoding + EOT


def test_parallel_token_chunks_matches_single_process(tmp_path):
    tok = FastTokenizer.train(["hello world", "the cat sat", "a b c d"] * 6,
                              vocab_size=300, save_path=str(tmp_path / "tok.json"))
    docs = [f"doc number {i} with a few words here" for i in range(20)]
    eot = tok.eot_token
    single = np.concatenate(list(batched_token_chunks(tok, docs, eot, batch_docs=4)))
    parallel = np.concatenate(list(parallel_token_chunks(
        tmp_path / "tok.json", docs, eot, workers=2, batch_docs=4)))
    assert parallel.tolist() == single.tolist()  # identical tokens, order preserved (imap)


def test_take_tokens_splits_without_loss_or_overlap():
    chunks = iter([np.arange(0, 400, dtype=np.uint16), np.arange(400, 1000, dtype=np.uint16)])
    carry: list = []
    val = np.concatenate(list(take_tokens(chunks, 500, carry)))
    train = np.concatenate(list(take_tokens(chunks, 500, carry)))
    assert val.tolist() == list(range(500))            # exactly 500, split mid-chunk
    assert train.tolist() == list(range(500, 1000))    # resumes with no loss/overlap


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
