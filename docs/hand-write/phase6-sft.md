> **Exercise — on `main`, no branch switching.** Implement the stub in `src/microlab/exercises/phase06_sft.py`, then run `pytest -m exercise -k phase06_sft` to grade it against the reference oracle. Your solution is tracked in git; commit it when it passes.

# START HERE — supervised fine-tuning: loss masking (Phase 6)

You're on the exercises folder on `main`. You implement two functions in
`src/microlab/exercises/phase06_sft.py`; the chat template, collator, and SFT training loop are
already on `main`. The differential tests grade you against `microlab.model.reference.sft`.

## 1. See SFT run on real instruction data first (~1 min)

Fine-tune the tiny GPT on real Dolly-15k instructions (git-ignored under `data/corpora/`):

```python
import torch
from microlab.data.reference.loaders import load_dolly
from microlab.tokenizer.reference.bpe import BPETokenizer
from microlab.model.reference.gpt import GPT, GPTConfig
from microlab.model.reference.sft import format_chat, build_sft_example, train_sft, IGNORE_INDEX
from microlab.model.reference.train import TrainConfig

rows = [r for r in load_dolly("data/corpora/dolly15k.jsonl", limit=3000)
        if not r["context"].strip() and len(r["instruction"]) + len(r["response"]) < 300][:400]
tok = BPETokenizer(); tok.train("\n".join(r["instruction"] + " " + r["response"] for r in rows), 512)
ex = []
for r in rows:
    p, resp = format_chat(r["instruction"], "", r["response"])
    inp, lab = build_sft_example(tok, p, resp)
    if len(inp) <= 128: ex.append((inp, lab))
m = GPT(GPTConfig(vocab_size=512, block_size=128, n_layer=4, n_head=4, n_embd=192))
print(train_sft(m, ex, TrainConfig(steps=600, batch_size=16, block_size=128, device="cuda"))["final_loss"])
```

Observed: ~42% of label positions are supervised (the responses; the prompt is masked),
masked loss falls 6.3 → 0.3, and generation produces a response continuation. Toy quality
at vocab-512 scale — but the masking machinery is exactly what a real SFT run uses.

## 2. What you implement

```bash
/home/rje/anaconda3/bin/conda run -n microlab pytest tests/exercises/test_phase06_sft.py -v
```

1. **`build_sft_example(tok, prompt_text, response_text)`** — return `input_ids` (encoded
   prompt then response) and `labels` (same length) where **prompt positions are
   `IGNORE_INDEX` and response positions are the response token ids**. This is the whole
   idea of SFT: don't train the model to reproduce the prompt, only to produce the response.
2. **`masked_cross_entropy(logits, labels)`** — the **causal shift**: `logits[:, :-1]`
   predict `labels[:, 1:]` (position t predicts token t+1), then `F.cross_entropy(...,
   ignore_index=IGNORE_INDEX)`. The shift is required because `build_sft_example` makes
   labels position-aligned with inputs (a masked copy), and the LM predicts the *next* token.

## 3. Why this matters

SFT turns a next-token predictor into an instruction follower by training on
(prompt → response) pairs. Masking the prompt is what makes it *supervised fine-tuning* and
not just more pretraining: the gradient only flows from the tokens you want the model to
generate. Get the mask wrong (supervise the prompt, or mis-align the shift) and you either
waste capacity memorizing prompts or corrupt the loss — both common real bugs.

## 4. Honest scale note

vocab-512 + 400 examples won't produce a real assistant; it proves the masking/loop are
correct. Real instruction following needs a real base model + full instruction set — same
code, bigger model and data (and a fast tokenizer, since our BPE won't scale to that).

## 5. When you're done

`pytest tests/exercises/test_phase06_sft.py` green → ping me for the Socratic review, then try
masking the prompt vs NOT masking it and compare — feeling why the mask matters is the lesson.
