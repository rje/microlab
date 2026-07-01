> **Exercise — on `main`, no branch switching.** Implement the stub in `src/microlab/exercises/phase10_rl.py`, then run `pytest -m exercise -k phase10_rl` to grade it against the reference oracle. Your solution is tracked in git; commit it when it passes.

# START HERE — RL on verifiable rewards: GRPO + PPO clip (Phase 10)

You're on the exercises folder on `main`. You implement the reward function, the group-normalized
advantage, and the PPO clipped loss in `src/microlab/exercises/phase10_rl.py`. `extract_answer`
(parsing "#### N" or the last number in a string) is already on `main` — reuse it, don't
reimplement it. Differential tests grade you against `microlab.model.reference.rl`.

## 1. See PPO-clip descend on synthetic advantages first (~1 min, needs a GPU)

```python
import torch
from microlab.model.reference.rl import ppo_clip_loss

torch.manual_seed(0)
logits = torch.randn(8, 32, requires_grad=True, device="cuda")
old = torch.log_softmax(logits.detach(), dim=-1)
actions = torch.randint(0, 32, (8,), device="cuda")
adv = torch.randn(8, device="cuda")
opt = torch.optim.SGD([logits], lr=0.1)
for _ in range(20):
    lp = torch.log_softmax(logits, dim=-1)[torch.arange(8, device="cuda"), actions]
    olp = old[torch.arange(8, device="cuda"), actions]
    loss = ppo_clip_loss(lp, olp, adv)
    opt.zero_grad(); loss.backward(); opt.step()
print(loss.item())  # falls monotonically
```

## 2. What you implement

```bash
/home/rje/anaconda3/bin/conda run -n microlab pytest tests/exercises/test_phase10_rl.py -v
```

1. **`verifiable_reward(generated, gold)`** — `1.0` if `extract_answer(generated) ==
   extract_answer(gold)` (and isn't `None`), else `0.0`. No learned reward model, no human
   labels — the environment itself checks the answer, which is what makes math/code RL scale
   cheaply (GSM8K, code execution, etc).
2. **`group_normalized_advantages(rewards, eps=1e-8)`** — `(rewards - mean) / (std + eps)`
   over one group (several completions sampled for the *same* prompt). This is GRPO's trick:
   skip the learned value-function baseline of classic PPO and use the group's own mean/std
   as the baseline instead.
3. **`ppo_clip_loss(logprobs, old_logprobs, advantages, clip_eps=0.2)`** — with
   `ratio = exp(logprobs - old_logprobs)`, the loss (to minimize) is
   `-mean(min(ratio*A, clip(ratio, 1-eps, 1+eps)*A))`. The clip removes the incentive to move
   the policy arbitrarily far from the policy that generated the samples in a single update.

## 3. Why this matters

This is the recipe behind reasoning-RL results (DeepSeek-R1-Zero style): sample several
completions per prompt, score each with a *verifiable* reward (does the final answer match?),
turn the group's rewards into advantages with no separate value network, and update the
policy with the same clipped surrogate PPO uses to stay stable off-policy. Group-relative
advantages are why GRPO needs no critic — the group IS the baseline.

## 4. How it's graded

Differential tests compare all three functions against `microlab.model.reference.rl` (same
GSM8K-style strings for the reward, random tensors for advantages/PPO), plus known-value
checks: mean-zero advantages, and the classic PPO-clip case `ratio=e, A=1, eps=0.2 → loss =
-1.2` (the clipped branch wins since `A>0` and `ratio > 1+eps`). Green → ping me for the
Socratic review.
