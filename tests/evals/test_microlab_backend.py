import torch

from microlab.evals.backends import MicrolabBackend
from microlab.evals.schema import EvalTask
from microlab.model.reference.gpt import GPT, GPTConfig


class _FakeTok:
    def encode(self, s):
        return [ord(c) % 64 for c in s] or [0]

    def decode(self, ids):
        return "".join(chr(65 + (i % 26)) for i in ids)


def test_microlab_backend_returns_completion():
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=2, n_embd=32))
    be = MicrolabBackend(m, _FakeTok(), max_new_tokens=8, temperature=0.0, device="cpu")
    task = EvalTask(id="t", category="story", prompt="hello", checks=[])
    out = be.generate(task)
    assert out.task_id == "t" and isinstance(out.text, str) and out.latency_seconds >= 0
