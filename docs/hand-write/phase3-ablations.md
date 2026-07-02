> **Exercise — on `main`, no branch switching.** Implement the stub in `src/microlab/exercises/phase03_variants.py`, then run `pytest -m exercise -k phase03_variants` to grade it against the reference oracle. Your solution is tracked in git; commit it when it passes.

# START HERE — hand-write the architecture variants (Phase 3)

You're on the exercises folder on `main`. You implement three primitives in
`src/microlab/exercises/phase03_variants.py`; the reference variants, the configurable
`VariantGPT`, and the ablation runner are already built and green on `main`. The
differential tests grade you against `microlab.model.reference.variants`.

## 1. See the ablation run on the GPU first (1 min)

```python
# python  (after: conda activate microlab)
import torch
from microlab.data.reference.loaders import load_sample
from microlab.data.reference.pipeline import clean_text
from microlab.tokenizer.reference.bpe import BPETokenizer
from microlab.model.reference.variants import VariantConfig
from microlab.model.reference.train import TrainConfig
from microlab.model.reference.ablate import run_ablations

text = clean_text(load_sample()) * 50
tok = BPETokenizer(); tok.train(text, 512)
data = torch.tensor(tok.encode(text), dtype=torch.long)
base = VariantConfig(vocab_size=512, block_size=128, n_layer=4, n_head=4, n_embd=192)
print(run_ablations(data, base, TrainConfig(steps=300, batch_size=32, block_size=128, device="cuda")))
```

Observed: RoPE drops the learned-position params (1.88M vs 1.90M), SwiGLU keeps params
comparable, all four train. That table — loss / params / throughput per variant — is the
whole point of the phase: change one thing, measure.

## 2. What you implement

```bash
/home/rje/anaconda3/bin/conda run -n microlab pytest tests/exercises/test_phase03_variants.py -v
```

Four pieces in `student_variants.py`:

1. **`RMSNorm.forward`** — `x / sqrt(mean(x², last dim) + eps) * weight`. No mean
   subtraction, no bias (that's the whole difference from LayerNorm). Graded vs the
   reference + a magnitude-normalization property.
2. **`apply_rope`** — rotary position embeddings on `x` of shape `(B, n_head, T, head_dim)`.
   Duplicate `cos`/`sin` to full `head_dim` (`cat((t, t), -1)`), and
   `out = x * cos_full + rotate_half(x) * sin_full` where `rotate_half(x) = cat(-x2, x1)`
   over the two halves. Graded vs the reference + a norm-preservation property (a rotation
   can't change a vector's length).
3. **`SwiGLUMLP.forward`** — `w2(silu(w1 x) * w3 x)` then dropout. The gated GLU variant;
   the `silu(w1 x)` branch gates the `w3 x` branch. Graded vs the reference (weights copied
   in, so the hidden-dim formula already matches).
4. **`GQAAttention.forward`** — grouped-query attention: `n_head` query heads share
   `n_kv_head` K/V heads. Project q normally; project kv at `2*n_kv_head*head_dim` and
   `.split(n_kv_head*head_dim, dim=2)`; RoPE on q,k; `repeat_interleave` k,v by
   `n_head//n_kv_head`; causal SDPA. `n_kv_head=1` is MQA (Shazeer 2019), `=n_head` is
   plain MHA. **Why it exists won't fully land until Phase 6**: the KV *cache* shrinks by
   `n_head/n_kv_head`, and at inference time the cache — not compute — is the bottleneck.
   Ablate it now (add `n_kv_head` to your ablation matrix: loss barely moves); measure the
   cache payoff when you build inference.

## 3. Why these three (the papers)

- **RMSNorm** — cheaper norm (no mean/centering); used in LLaMA/T5. Question to hold:
  why does dropping the mean-subtraction barely hurt?
- **RoPE** — encodes *relative* position by rotating q/k, so attention scores depend on
  `m − n`. Why does that generalize to longer contexts better than a learned table?
- **SwiGLU** — a gated MLP that consistently beats GELU at equal params (Shazeer's "GLU
  Variants Improve Transformer"). The hidden dim is `8/3·n_embd` so the gate's extra matrix
  doesn't blow up the param count.

## 4. When you're done

`pytest tests/exercises/test_phase03_variants.py` green → ping me for the Socratic review, then
run your own variants through `run_ablations` on the GPU and see which actually helps on
your corpus.

## Optional stretch

- Run the ablation for more steps on TinyStories and compare *val* loss, not train loss.
- Combine all three (rms + rope + swiglu) — that's essentially the LLaMA block. Does the
  combination beat the sum of the individual gains?
