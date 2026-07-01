> **Exercise — on `main`, no branch switching.** Implement the stub in `src/microlab/exercises/phase08_reward.py`, then run `pytest -m exercise -k phase08_reward` to grade it against the reference oracle. Your solution is tracked in git; commit it when it passes.

# START HERE — reward modeling: Bradley-Terry preference loss (Phase 8)

You're on the exercises folder on `main`. You implement the pairwise preference loss and accuracy
metric in `src/microlab/exercises/phase08_reward.py`; the `RewardModel` (GPT + scalar value head)
is already on `main`. Differential tests grade you against `microlab.model.reference.reward`.

## 1. See a reward model learn a preference first (~1 min, needs a GPU)

```python
import torch
from microlab.model.reference.gpt import GPT, GPTConfig
from microlab.model.reference.reward import RewardModel, preference_loss

torch.manual_seed(0)
rm = RewardModel(GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=64))).cuda()
opt = torch.optim.AdamW(rm.parameters(), lr=1e-3)
chosen = torch.randint(0, 64, (8, 16), device="cuda")
rejected = torch.randint(0, 64, (8, 16), device="cuda")
for _ in range(100):
    loss = preference_loss(rm.sequence_reward(chosen), rm.sequence_reward(rejected))
    opt.zero_grad(); loss.backward(); opt.step()
print(loss.item())  # falls well below log(2) ~ 0.69 as the model learns to separate them
```

Random token sequences carry no real preference signal, so this just proves the plumbing —
the loss pushes `reward(chosen)` above `reward(rejected)` regardless of what "chosen" means.

## 2. What you implement

```bash
/home/rje/anaconda3/bin/conda run -n microlab pytest tests/exercises/test_phase08_reward.py -v
```

1. **`preference_loss(chosen_rewards, rejected_rewards)`** — Bradley-Terry:
   `-log sigmoid(r_chosen - r_rejected)`, averaged over the batch. This is the same pairwise
   ranking loss under a logistic model of "probability chosen is preferred."
2. **`reward_accuracy(chosen_rewards, rejected_rewards)`** — fraction of pairs where
   `r_chosen > r_rejected`. A cheap sanity metric alongside the loss: loss can keep falling
   long after accuracy saturates at 1.0.

## 3. Why this matters

RLHF (and DPO, next phase) starts from human preference pairs: given two responses to the
same prompt, which one do people prefer? Bradley-Terry turns those binary comparisons into a
scalar reward model by assuming `P(chosen ≻ rejected) = sigmoid(r_chosen - r_rejected)` —
maximizing that likelihood is exactly minimizing this loss. Get this wrong (e.g. swap
chosen/rejected, or forget the mean) and the reward model silently rewards the wrong
behavior — a famously hard-to-detect RLHF failure mode.

## 4. How it's graded

Differential tests in `tests/exercises/test_phase08_reward.py` compare your output
element-for-element against `microlab.model.reference.reward` on random reward tensors, plus
a known-value check: equal rewards give exactly `log(2)` loss (`sigmoid(0) = 0.5`). Green →
ping me for the Socratic review.
