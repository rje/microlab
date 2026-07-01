> **Exercise — on `main`, no branch switching.** Implement the stub in `src/microlab/exercises/phase05_continued.py`, then run `pytest -m exercise -k phase05_continued` to grade it against the reference oracle. Your solution is tracked in git; commit it when it passes.

# START HERE — continued pretraining: forgetting + replay (Phase 5)

You're on the exercises folder on `main`. You implement two small functions in
`src/microlab/exercises/phase05_continued.py`; the continued-pretraining runner and the
per-corpus evaluator are already on `main`. The differential test grades you against
`microlab.model.reference.continued`.

## 1. See the phenomenon on the GPU first (~1 min)

Continue-train a Shakespeare model on Sherlock Holmes and watch it forget — then watch
replay fix it (both corpora are git-ignored under `data/corpora/`; fetch via the loaders):

```python
import copy, torch
from microlab.data.reference.loaders import load_tinyshakespeare, load_text_file
from microlab.data.reference.pipeline import clean_text
from microlab.tokenizer.reference.bpe import BPETokenizer
from microlab.model.reference.gpt import GPT, GPTConfig
from microlab.model.reference.train import TrainConfig, train
from microlab.model.reference.continued import continued_pretrain

shake = clean_text(load_tinyshakespeare("data/corpora/tinyshakespeare.txt"))[:300_000]
sher  = clean_text(load_text_file("data/corpora/sherlock.txt"))[:300_000]
tok = BPETokenizer(); tok.train(shake + sher, 512)
enc = lambda t: torch.tensor(tok.encode(t), dtype=torch.long)
sh, sl = enc(shake), enc(sher)
base = GPT(GPTConfig(vocab_size=512, block_size=64, n_layer=2, n_head=2, n_embd=64))
train(base, sh[:int(len(sh)*.9)], TrainConfig(steps=800, batch_size=32, block_size=64, device="cuda"))
corpora = {"old": sh[int(len(sh)*.9):], "new": sl[int(len(sl)*.9):]}
cfg = TrainConfig(steps=400, batch_size=32, block_size=64, device="cuda")
for frac, replay in [(0.0, None), (0.25, sh)]:
    print(frac, continued_pretrain(copy.deepcopy(base), sl, corpora, cfg, replay_data=replay, replay_fraction=frac)["forgetting"])
```

Observed: no replay → old-domain loss rises ~+0.47 (forgetting); 25% replay → ~+0.23
(roughly halved) while still learning the new domain. That trade-off IS the phase.

## 2. What you implement

```bash
/home/rje/anaconda3/bin/conda run -n microlab pytest tests/exercises/test_phase05_continued.py -v
```

1. **`forgetting_score(loss_before, loss_after)`** — how much the original-domain loss went
   up after continued training. One line, but it's the metric the whole phase turns on.
2. **`build_replay_mix(new_tokens, old_tokens, replay_fraction)`** — mix in a share of old
   tokens so training rehearses the old domain. To make old tokens a fraction `f` of the
   total you need `n_old = f/(1-f) · n_new` of them; cap at the old corpus size; `f == 0`
   returns the new tokens unchanged; `f` must be `< 1`.

## 3. Why this matters (the papers / practice)

Continued/domain-adaptive pretraining is how you specialize a base model without paying to
train from scratch — but naive continuation causes **catastrophic forgetting** of the
original capabilities. **Replay** (rehearsing a slice of the original data) is the simplest
mitigation; regularization (EWC) and parameter-isolation (adapters — Phase 7) are others.
The measurement you're building — loss on old vs new — is exactly how you'd tune the
replay ratio for a real run.

## 4. Honest scale note

At vocab-512 / 0.14M-param toy scale the *numbers* are small, but the *effect* is real and
reproducible (above). On a real base model + corpus the same code measures real forgetting;
only the data and model size change.

## 5. When you're done

`pytest tests/exercises/test_phase05_continued.py` green → ping me for the Socratic review, then
sweep the replay fraction (0, 0.1, 0.25, 0.5) and plot forgetting-vs-learning to find the
knee.
