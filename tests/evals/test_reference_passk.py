from microlab.evals.reference.passk import passk_eval
from microlab.evals.schema import Check, EvalTask, ModelOutput


class CountingBackend:
    """Returns the correct answer for the first `n_correct` calls, then a wrong one."""

    def __init__(self, n_correct):
        self.n_correct = n_correct
        self.calls = 0

    def generate(self, task):
        self.calls += 1
        text = "4" if self.calls <= self.n_correct else "nope"
        return ModelOutput(task_id=task.id, text=text, latency_seconds=0.0)


def _task():
    return EvalTask(id="t", category="math", prompt="Answer 4.",
                    checks=[Check(type="exact_match", value="4")])


def test_passk_all_correct():
    res = passk_eval(CountingBackend(10), _task(), n_samples=10, ks=[1, 5])
    assert res[1] == 1.0 and res[5] == 1.0


def test_passk_none_correct():
    res = passk_eval(CountingBackend(0), _task(), n_samples=10, ks=[1, 10])
    assert res[1] == 0.0 and res[10] == 0.0


def test_passk_partial_matches_formula():
    from microlab.evals.reference.metrics import pass_at_k
    res = passk_eval(CountingBackend(3), _task(), n_samples=10, ks=[1, 5])
    assert res[1] == pass_at_k(10, 3, 1)
    assert res[5] == pass_at_k(10, 3, 5)
