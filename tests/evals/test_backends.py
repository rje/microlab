from microlab.evals.backends import FixtureBackend, OllamaBackend, create_backend
from microlab.evals.schema import EvalTask


def test_fixture_backend_returns_configured_answer():
    backend = FixtureBackend({"task-001": "4"})
    task = EvalTask(
        id="task-001",
        category="math",
        prompt="Answer only with 4.",
        checks=[],
    )

    output = backend.generate(task)

    assert output.task_id == "task-001"
    assert output.text == "4"
    assert output.latency_seconds >= 0


def test_create_backend_builds_fixture_backend():
    backend = create_backend({"type": "fixture", "answers": {"task-001": "4"}})
    assert isinstance(backend, FixtureBackend)


def test_create_backend_builds_ollama_backend():
    backend = create_backend(
        {
            "type": "ollama",
            "model": "qwen3.6:27b",
            "host": "http://127.0.0.1:11434",
            "temperature": 0,
        }
    )

    assert isinstance(backend, OllamaBackend)
    assert backend.model == "qwen3.6:27b"
