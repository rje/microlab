> **Exercise — on `main`, no branch switching.** Implement the stub in `src/microlab/exercises/phase14_reasoning.py`, then run `pytest -m exercise -k phase14_reasoning` to grade it against the reference oracle. Your solution is tracked in git; commit it when it passes.

# START HERE — STaR bootstrapping + knowledge distillation (Phase 14)

You're on the exercises folder on `main`. You implement the STaR trace filter and the distillation
loss in `src/microlab/exercises/phase14_reasoning.py`; `self_consistency` (majority vote) is
already on `main`. Differential tests grade you against `microlab.model.reference.reasoning`.

## 1. See distillation pull a student toward a frozen teacher first (~1 min, needs a GPU)

```python
import torch
from microlab.model.reference.gpt import GPT, GPTConfig
from microlab.model.reference.reasoning import distillation_loss

torch.manual_seed(0)
cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=64)
teacher = GPT(cfg).cuda().eval()
student = GPT(cfg).cuda()
opt = torch.optim.AdamW(student.parameters(), lr=1e-3)
x = torch.randint(0, 64, (4, 16), device="cuda")
with torch.no_grad():
    tlog = teacher(x)[0]
for _ in range(60):
    loss = distillation_loss(student(x)[0], tlog)
    opt.zero_grad(); loss.backward(); opt.step()
print(loss.item())  # falls toward 0 as student matches teacher's distribution
```

## 2. What you implement

```bash
/home/rje/anaconda3/bin/conda run -n microlab pytest tests/exercises/test_phase14_reasoning.py -v
```

1. **`filter_correct_traces(traces, gold, extract_fn)`** — keep only the traces whose
   `extract_fn(trace)` equals `gold` (and isn't `None`). This is the entire STaR loop in one
   line: sample many reasoning traces, throw away the ones that got the wrong answer, and
   fine-tune on the survivors — the model bootstraps from its own successes, no human-written
   rationales needed.
2. **`distillation_loss(student_logits, teacher_logits, temperature=2.0)`** —
   `KL(teacher || student)` on `T`-softened distributions (`log_softmax(student/T)`,
   `softmax(teacher/T)`), scaled by `T²` to keep gradient magnitude roughly constant across
   choices of `T` (the classic Hinton et al. correction). Zero exactly when
   `student_logits == teacher_logits`.

## 3. Why this matters

STaR (Zelikman et al.) is a cheap way to get a model to reason better *without* an RL loop or
a reward model: sample rationales, keep only the ones that check out, fine-tune on those (loop
if you like). Distillation is the complementary move at deployment time — once you've grown a
strong (possibly huge, possibly RL-trained) teacher, you compress its behavior into a small
student by matching softened output distributions rather than hard labels, which transfers
more of the teacher's "uncertainty structure" than one-hot cross-entropy would.

## 4. How it's graded

Differential tests compare `filter_correct_traces` against the reference on a small toy
trace/gold set with a couple of `extract_fn`s, and `distillation_loss` against the reference
across several temperatures, plus a known-value check: `student_logits == teacher_logits`
gives exactly `0.0`. Green → ping me for the Socratic review.
