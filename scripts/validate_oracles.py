"""Validate the reference oracles on a larger real corpus with a proper train/val split
and held-out validation loss. Exercises the tokenizer, model+training, ablation runner,
and scaling sweep at ~100x the bundled-sample scale, so the scaling/ablation numbers are
meaningful (val loss) rather than memorization artifacts.

    python scripts/validate_oracles.py [path-to-text] [n_chars]

Default corpus: data/corpora/tinyshakespeare.txt (public domain).
"""

from __future__ import annotations

import sys
import time

import torch

from microlab.data.reference.loaders import load_tinyshakespeare
from microlab.data.reference.pipeline import clean_text
from microlab.model.reference.ablate import run_ablations
from microlab.model.reference.gpt import GPT, GPTConfig
from microlab.model.reference.sample import generate
from microlab.model.reference.scaling import run_scaling_sweep
from microlab.model.reference.train import TrainConfig, train
from microlab.model.reference.variants import VariantConfig
from microlab.tokenizer.reference.bpe import BPETokenizer

PATH = sys.argv[1] if len(sys.argv) > 1 else "data/corpora/tinyshakespeare.txt"
N_CHARS = int(sys.argv[2]) if len(sys.argv) > 2 else 300_000
VOCAB, BLOCK = 512, 128


def timed(label, fn):
    s = time.time()
    r = fn()
    print(f"[{time.time() - s:6.1f}s] {label}")
    return r


text = clean_text(load_tinyshakespeare(PATH))[:N_CHARS]
split = int(len(text) * 0.9)
train_text, val_text = text[:split], text[split:]
print(f"corpus: total={len(text)} train={len(train_text)} val={len(val_text)} chars")

# --- 1. tokenizer at scale: train on TRAIN only (no val leakage) ---
tok = BPETokenizer()
timed("train BPE", lambda: tok.train(train_text, VOCAB))
train_ids = timed("encode train", lambda: tok.encode(train_text))
val_ids = timed("encode val", lambda: tok.encode(val_text))
roundtrip = tok.decode(train_ids) == train_text
comp = len(train_text.encode()) / len(train_ids)
print(f"TOKENIZER: round_trip_exact={roundtrip} compression={comp:.2f} bytes/token vocab={VOCAB}")

train_data = torch.tensor(train_ids, dtype=torch.long)
val_data = torch.tensor(val_ids, dtype=torch.long)

# --- 2. one model: train + val loss + a text sample ---
model = GPT(GPTConfig(vocab_size=VOCAB, block_size=BLOCK, n_layer=4, n_head=4, n_embd=256))
stats = timed(
    "train model",
    lambda: train(
        model, train_data,
        TrainConfig(steps=2000, batch_size=32, block_size=BLOCK, device="cuda"),
        val_data=val_data,
    ),
)
print(
    f"MODEL: params={model.num_params() / 1e6:.2f}M train_loss={stats['history'][-1]:.3f} "
    f"val_loss={stats['val_loss']:.3f} tok/s={stats['tokens_per_sec']:.0f} "
    f"vram={stats['peak_vram_mb']:.0f}MB"
)
device = stats["device"]
start = torch.tensor([tok.encode("KING RICHARD")], dtype=torch.long, device=device)
sample = tok.decode(generate(model.to(device), start, 220, temperature=0.8, top_k=40)[0].tolist())
print("--- sample ---\n" + sample + "\n---")

# --- 3. ablation on VAL loss ---
base = VariantConfig(vocab_size=VOCAB, block_size=BLOCK, n_layer=4, n_head=4, n_embd=256)
abl = timed(
    "ablation",
    lambda: run_ablations(
        train_data, base,
        TrainConfig(steps=1500, batch_size=32, block_size=BLOCK, device="cuda"),
        val_data=val_data,
    ),
)
print("ABLATION (held-out val loss):")
for name, r in abl.items():
    print(
        f"  {name:<9} params={r['params'] / 1e6:.2f}M "
        f"train={r['final_loss']:.3f} val={r['val_loss']:.3f}"
    )

# --- 4. scaling sweep on VAL loss ---
sweep = timed(
    "scaling sweep",
    lambda: run_scaling_sweep(
        train_data, [64, 128, 256, 384],
        TrainConfig(steps=1500, batch_size=32, block_size=BLOCK, device="cuda"),
        val_data=val_data, vocab_size=VOCAB,
    ),
)
print(
    f"SCALING (fit_on={sweep['fit_on']}): loss ~ N^({-sweep['alpha']:+.3f}) "
    f"(A={sweep['A']:.2f}; a negative exponent means loss falls as models grow — the healthy sign)"
)
for p in sweep["points"]:
    print(
        f"  n_embd={p['n_embd']:>4} params={p['params'] / 1e6:>6.2f}M "
        f"train={p['loss']:.3f} val={p['val_loss']:.3f}"
    )
