> **Exercise — on `main`, no branch switching.** Implement the stub in `src/microlab/exercises/phase01_bpe.py`, then run `pytest -m exercise -k phase01_bpe` to grade it against the reference oracle. Your solution is tracked in git; commit it when it passes.

# START HERE — hand-write a byte-level BPE tokenizer (Phase 1)

You're on branch the exercises folder on `main`. You implement one file:
`src/microlab/exercises/phase01_bpe.py` (three methods, currently `NotImplementedError`).
The reference oracle, the data pipeline, and the GPU dataset are already built and
green on `main` — you only write the tokenizer.

## 1. See the pieces that already exist (2 min)

```bash
# the data pipeline + bundled sample corpus the tokenizer will train on
python -c "from microlab.data.reference.loaders import load_sample; print(load_sample()[:200])"
```

The reference oracle lives at `src/microlab/tokenizer/reference/bpe.py`. Don't open it
until you've attempted — the tests use it to grade you.

## 2. What you implement

`BPETokenizer` with `train(text, vocab_size)`, `encode(text) -> list[int]`,
`decode(ids) -> str`. Run the spec:

```bash
/home/rje/anaconda3/bin/conda run -n microlab pytest tests/exercises/test_phase01_bpe.py -v
```

Red until you implement. The tests: round-trip on arbitrary unicode, untrained =
identity bytes, vocab grows to target, deterministic training, encode uses merges, and
a **differential check against the reference oracle**.

## 3. The algorithm (byte-level BPE — GPT-2 / minBPE style)

Work in **bytes**, not characters — that's what makes `decode(encode(x)) == x` hold for
any unicode (and why `vocab` maps ids → `bytes`).

**train(text, vocab_size):**
1. `ids = list(text.encode("utf-8"))` — start as raw bytes (0–255).
2. Repeat for new ids `256, 257, … vocab_size-1`:
   - count adjacent pairs `(ids[i], ids[i+1])`,
   - pick the **most frequent** pair — on ties, the one with the larger byte values
     (`argmax` over `(count, -a, -b)`; this fixed tie-break is what makes your merges
     match the oracle exactly),
   - replace every occurrence of that pair with the new id,
   - record `merges[pair] = new_id` and `vocab[new_id] = vocab[a] + vocab[b]`.
   - Stop early if there are no pairs left (a small corpus can exhaust).

**encode(text):** start from utf-8 bytes; repeatedly find the present pair with the
**lowest merge id** (earliest learned) and apply it, until no learned pair remains.

**decode(ids):** `b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")`.

## 4. GPU note (looking ahead to Phase 2)

The tokenizer is CPU work, but it feeds the GPU: `microlab.data.reference.dataset.get_batch`
turns a token tensor into `(x, y)` blocks and moves them to CUDA with pinned-memory
non-blocking transfer. When you train the Phase-2 model you'll tokenize a corpus once,
keep the ids as a tensor, and sample batches from it on the GPU.

## 5. When you're done

`pytest tests/exercises/test_phase01_bpe.py` all green → ping me for the Socratic review, then we
promote your tokenizer or keep it beside the reference. The differential test means green
= byte-for-byte agreement with the oracle, including the tie-break.

## Optional stretch

- Add a `save`/`load` (merges + vocab to JSON) so a trained tokenizer is reusable.
- Add a regex pre-split (GPT-2 splits on a pattern before BPE) and measure the change in
  bytes/token compression on the sample corpus.
