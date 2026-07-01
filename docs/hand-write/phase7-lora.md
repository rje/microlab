> **Exercise — on `main`, no branch switching.** Implement the stub in `src/microlab/exercises/phase07_lora.py`, then run `pytest -m exercise -k phase07_lora` to grade it against the reference oracle. Your solution is tracked in git; commit it when it passes.

# START HERE — efficient fine-tuning: LoRA + QLoRA (Phase 7)

You're on the exercises folder on `main`. You implement the LoRA adapter math and a QLoRA-style
quantizer in `src/microlab/exercises/phase07_lora.py`; the `apply_lora_to_gpt` wiring, the
param counters, and the training path are already on `main`. Differential tests grade you
against `microlab.model.reference.lora`.

## 1. See LoRA fine-tuning on the GPU first (~1 min)

Pretrain on Shakespeare, then adapt to Sherlock Holmes training **only ~5% of the params**:

```python
import torch
from microlab.data.reference.loaders import load_tinyshakespeare, load_text_file
from microlab.data.reference.pipeline import clean_text
from microlab.tokenizer.reference.bpe import BPETokenizer
from microlab.model.reference.gpt import GPT, GPTConfig
from microlab.model.reference.train import TrainConfig, train, estimate_loss
from microlab.model.reference.lora import apply_lora_to_gpt, count_trainable, count_total

shake = clean_text(load_tinyshakespeare("data/corpora/tinyshakespeare.txt"))[:300_000]
sher  = clean_text(load_text_file("data/corpora/sherlock.txt"))[:300_000]
tok = BPETokenizer(); tok.train(shake + sher, 512)
sh = torch.tensor(tok.encode(shake)); sl = torch.tensor(tok.encode(sher))
m = GPT(GPTConfig(vocab_size=512, block_size=64, n_layer=4, n_head=4, n_embd=192))
train(m, sh, TrainConfig(steps=600, batch_size=32, block_size=64, device="cuda"))
apply_lora_to_gpt(m, rank=8, alpha=16)
print(f"trainable {count_trainable(m)/count_total(m)*100:.1f}%")
train(m, sl[:int(len(sl)*.9)], TrainConfig(steps=600, batch_size=32, block_size=64, device="cuda"))
```

Observed: LoRA trains **4.9%** of params, Sherlock val loss falls 4.58 → 4.04, and the merged
weights reproduce the adapter exactly (fold-in for zero inference overhead).

## 2. What you implement

```bash
/home/rje/anaconda3/bin/conda run -n microlab pytest tests/exercises/test_phase07_lora.py -v
```

1. **`LoRALinear.forward`** — `base(x) + scaling * (x @ A^T @ B^T)`. The `__init__` is given
   (A random, **B zero** → the adapter is a no-op at init, so the wrapped layer starts exactly
   equal to the base). Shapes: `A` is `(rank, in)`, `B` is `(out, rank)`.
2. **`LoRALinear.merged_weight`** — `base.weight + scaling * (B @ A)`. Folding the low-rank
   update into a full weight matrix; a plain Linear with this weight equals the adapter's
   output (that's what merge does for deployment).
3. **`quantize_dequantize(w, bits)`** — symmetric absmax: `scale = max(|w|)/(2^(bits-1)-1)`,
   round `w/scale`, clamp to the int range, multiply back. Return `w` unchanged if `scale==0`.
   This is the QLoRA idea in miniature — quantize the frozen base to low-bit to save VRAM,
   keep the adapters in full precision.

## 3. Why this matters (the papers)

**LoRA** (Hu et al.) makes fine-tuning a big model cheap: freeze the base, learn a tiny
low-rank update per weight matrix — a few % of the params, a small optimizer-state footprint,
and *zero* extra inference cost once merged. **QLoRA** (Dettmers et al.) goes further:
4-bit-quantize the frozen base (NF4) so a large model fits on one GPU, and train adapters on
top. Together they're why a single RTX 6000 can fine-tune models far bigger than full
fine-tuning would allow. (Production QLoRA uses bitsandbytes NF4; our absmax quantizer is the
concept, not the exact scheme.)

## 4. When you're done

`pytest tests/exercises/test_phase07_lora.py` green → ping me for the Socratic review, then sweep
the rank (1, 2, 4, 8, 16) and watch the adaptation quality vs trainable-param trade-off.
