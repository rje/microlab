> **Exercise — on `main`, no branch switching.** Implement the stub in `src/microlab/exercises/phase09_dpo.py`, then run `pytest -m exercise -k phase09_dpo` to grade it against the reference oracle. Your solution is tracked in git; commit it when it passes.

# START HERE — Direct Preference Optimization (Phase 9)

You're on the exercises folder on `main`. You implement the sequence log-prob helper and the DPO loss in
`src/microlab/exercises/phase09_dpo.py`; `main` already has the GPT causal LM these numbers come
from. Differential tests grade you against `microlab.model.reference.dpo`.

## 1. See DPO reduce loss on paired responses first (~1 min, needs a GPU)

```python
import copy, torch
from microlab.model.reference.gpt import GPT, GPTConfig
from microlab.model.reference.dpo import sequence_logprob, dpo_loss

torch.manual_seed(0)
cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=64)
policy = GPT(cfg).cuda()
ref = copy.deepcopy(policy).eval()
opt = torch.optim.AdamW(policy.parameters(), lr=1e-3)
chosen = torch.randint(0, 64, (4, 16), device="cuda")
rejected = torch.randint(0, 64, (4, 16), device="cuda")
for _ in range(60):
    pc = sequence_logprob(policy(chosen)[0], chosen)
    pr = sequence_logprob(policy(rejected)[0], rejected)
    with torch.no_grad():
        rc = sequence_logprob(ref(chosen)[0], chosen)
        rr = sequence_logprob(ref(rejected)[0], rejected)
    loss, acc = dpo_loss(pc, pr, rc, rr, beta=0.1)
    opt.zero_grad(); loss.backward(); opt.step()
print(loss.item(), acc)  # loss falls from log(2), acc -> 1.0
```

DPO needs no reward model and no RL loop — it's a single classification-style loss computed
directly from two models' log-probs on the same (chosen, rejected) pairs.

## 2. What you implement

```bash
/home/rje/anaconda3/bin/conda run -n microlab pytest tests/exercises/test_phase09_dpo.py -v
```

1. **`sequence_logprob(logits, labels)`** — sum the per-token log-probs of `labels` under
   `logits`, over positions where `labels != IGNORE_INDEX` (-100). **Causal shift**:
   `logits[:, :-1]` predicts `labels[:, 1:]` (position t predicts token t+1) — the same shift
   as `masked_cross_entropy` in Phase 6, but summed instead of averaged, and returning one
   scalar per sequence (shape `(B,)`) instead of a batch mean.
2. **`dpo_loss(policy_chosen_logp, policy_rejected_logp, ref_chosen_logp, ref_rejected_logp,
   beta)`** — form the policy log-ratio `pi_chosen - pi_rejected` and the reference log-ratio
   `ref_chosen - ref_rejected`; the loss is `-log sigmoid(beta * (policy_ratio -
   ref_ratio))`, averaged. Also return the "implicit reward" accuracy: fraction of pairs
   where `beta*(pi_chosen - ref_chosen) > beta*(pi_rejected - ref_rejected)`.

## 3. Why this matters

DPO (Rafailov et al.) shows that RLHF's reward-model-then-PPO pipeline (Phase 8, then an RL
loop) is mathematically equivalent, under the Bradley-Terry model, to directly optimizing the
policy against a frozen copy of itself on preference pairs — no reward model, no rollouts, no
PPO instability. `beta` controls how far the policy is allowed to diverge from the reference;
the log-ratio-of-log-ratios is what makes this differentiable end-to-end from raw logits.

## 4. How it's graded

Differential tests compare `sequence_logprob` against the reference on random logits with
scattered `IGNORE_INDEX` masks, and `dpo_loss` against the reference on random log-prob
tensors, plus known-value checks: a hand-computed `sequence_logprob` example, and
`policy == ref` on all four DPO inputs collapsing the log-ratio to 0 for a loss of exactly
`log(2)`. Green → ping me for the Socratic review.
