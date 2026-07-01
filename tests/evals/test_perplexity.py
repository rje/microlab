import math

import torch

from microlab.evals.perplexity import evaluate_perplexity
from microlab.model.reference.gpt import GPT, GPTConfig


def test_perplexity_is_finite_and_gt_one():
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32))
    data = torch.randint(0, 64, (5000,))
    ppl = evaluate_perplexity(m, data, block_size=16, batch_size=8, iters=10)
    assert math.isfinite(ppl) and ppl > 1.0


def test_perplexity_drops_after_overfitting():
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=32, block_size=16, n_layer=2, n_head=2, n_embd=32))
    data = torch.randint(0, 32, (2000,))
    before = evaluate_perplexity(m, data, 16, 8, iters=10)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(200):
        # Train on the SAME objective evaluate_perplexity measures (next-token
        # prediction, y = x shifted by one) -- not an identity/copy task, which
        # would optimize an unrelated objective and needn't lower perplexity.
        x = data[:16 * 8].view(8, 16)
        y = data[1:16 * 8 + 1].view(8, 16)
        _, loss = m(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = evaluate_perplexity(m, data[:128], 16, 8, iters=10)
    assert after < before  # learned -> lower perplexity
