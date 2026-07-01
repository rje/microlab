> **Exercise — on `main`, no branch switching.** Implement the stub in `src/microlab/exercises/phase02_gpt.py`, then run `pytest -m exercise -k phase02_gpt` to grade it against the reference oracle. Your solution is tracked in git; commit it when it passes.

# START HERE — hand-write the tiny GPT core (Phase 2)

You're on the exercises folder on `main`. You implement four things in
`src/microlab/exercises/phase02_gpt.py`; the reference GPT, training loop, sampler, tokenizer,
and data pipeline are already built and green on `main`. The differential tests copy the
reference's weights into your modules and compare outputs, so green = your math matches
the oracle.

## 1. See the base run on the GPU first (1 min)

```bash
/home/rje/anaconda3/bin/conda run -n microlab python scripts/train_tiny_gpt.py
# tokenize -> train the reference GPT on CUDA -> sample text
# observed: loss 6.28 -> 0.03, ~0.9M tok/s, ~190MB VRAM, coherent on-corpus text
```

That's the whole pipeline you're plugging into. The reference oracle is
`src/microlab/model/reference/{gpt,train,sample}.py` — don't open it until you've tried.

## 2. What you implement

```bash
/home/rje/anaconda3/bin/conda run -n microlab pytest tests/exercises/test_phase02_gpt.py -v
```

Four `NotImplementedError` bodies in `student.py`:

1. **`StudentCausalSelfAttention.forward`** — multi-head causal self-attention.
   Project `x -> q,k,v` (`self.c_attn`), split into `n_head` heads, scaled dot-product
   `softmax(qkᵀ/√head_dim + causal_mask) v`, recombine heads, output projection
   (`self.c_proj`). The reference uses `F.scaled_dot_product_attention(is_causal=True)`;
   a manual implementation matches it to ~1e-4 — writing the mask/softmax by hand once is
   the point.
2. **`StudentBlock.forward`** — pre-norm residual block: `x = x + attn(ln_1(x))` then
   `x = x + mlp(ln_2(x))`. (It reuses the reference `MLP`, so only this wiring is yours.)
3. **`train_step`** — one optimization step: `zero_grad`, forward to the loss, `backward`,
   `optimizer.step()`, return the loss as a float. Graded by overfit-a-batch.
4. **`generate`** — the autoregressive sampling loop: crop context to `block_size`, take
   last-step logits, argmax when `temperature == 0` else top-k + temperature softmax +
   `multinomial`, append, repeat.

## 3. GPU idiosyncrasies to feel (the reason you wanted GPUs)

When you move past the CPU tests and run on CUDA (adapt `scripts/train_tiny_gpt.py` or
the reference `train()`), notice:
- **device & dtype:** inputs and model must be on the same device; `bf16` **autocast**
  wraps the forward/loss but params stay fp32 (mixed precision). Mismatches throw — read
  the error, it names the device/dtype.
- **honest timing:** CUDA is async; the reference calls `torch.cuda.synchronize()` before
  measuring tok/s. Without it your throughput numbers are fiction.
- **memory:** `torch.cuda.max_memory_allocated()` is the peak; if you OOM, lower
  `batch_size`/`block_size` or use `grad_accum` to keep the effective batch.
- **nondeterminism:** exact bitwise reproducibility on GPU isn't guaranteed — which is
  why the differential tests run on CPU and the GPU tests assert behavior (loss drops, no
  OOM), not equality.

## 4. When you're done

`pytest tests/exercises/test_phase02_gpt.py` all green → ping me for the Socratic review. Then run
the full GPU training and watch your own attention/block/loop drive the loss down on
hardware.

## Optional stretch

- Swap `F.scaled_dot_product_attention` back in and confirm your manual version agrees.
- Add KV-caching to `generate` and measure the speedup on a longer sample.
- Profile one training step with `torch.profiler` and find where the time goes.
