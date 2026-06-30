"""Short real GPU training run that ties the whole Phase-2 base together: tokenize a
corpus with the reference BPE, train the reference tiny GPT on CUDA (bf16), log
loss / peak VRAM / throughput, and sample text. Proves the base works end-to-end on
hardware.

    python scripts/train_tiny_gpt.py

Uses the bundled sample corpus by default (self-contained). For a longer run, point it
at TinyShakespeare / TinyStories via the loaders in microlab.data.reference.loaders.
"""

from __future__ import annotations

import torch

from microlab.data.reference.loaders import load_sample
from microlab.data.reference.pipeline import clean_text
from microlab.model.reference.gpt import GPT, GPTConfig
from microlab.model.reference.sample import generate
from microlab.model.reference.train import TrainConfig, train
from microlab.tokenizer.reference.bpe import BPETokenizer


def main() -> None:
    text = clean_text(load_sample()) * 50  # ~130 KB — enough tokens for a real run
    vocab_size = 512
    tok = BPETokenizer()
    tok.train(text, vocab_size=vocab_size)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"corpus: chars={len(text)} tokens={len(ids)} vocab={vocab_size}")

    block_size = 128
    model = GPT(
        GPTConfig(vocab_size=vocab_size, block_size=block_size, n_layer=4, n_head=4, n_embd=192)
    )
    print(f"model: params={model.num_params() / 1e6:.2f}M cuda={torch.cuda.is_available()}")

    stats = train(
        model,
        ids,
        TrainConfig(steps=400, batch_size=32, block_size=block_size, lr=3e-4, device="cuda"),
    )
    print(
        f"train: device={stats['device']} "
        f"loss {stats['history'][0]:.3f} -> {stats['final_loss']:.3f} "
        f"peak_vram={stats['peak_vram_mb']:.0f}MB tok/s={stats['tokens_per_sec']:.0f}"
    )

    device = stats["device"]
    start = torch.tensor([tok.encode("The ")], dtype=torch.long, device=device)
    out = generate(model.to(device), start, max_new_tokens=120, temperature=0.8, top_k=40)
    print("--- sample ---")
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
