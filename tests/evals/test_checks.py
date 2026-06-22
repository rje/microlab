from microlab.evals.checks import score_text
from microlab.evals.schema import Check, EvalTask, ModelOutput


def make_task(checks):
    return EvalTask(
        id="task-001",
        category="test",
        prompt="Prompt",
        checks=checks,
    )


def make_output(text):
    return ModelOutput(task_id="task-001", text=text, latency_seconds=0.01)


def test_exact_match_normalizes_whitespace():
    results = score_text(
        make_task([Check(type="exact_match", value="hello world")]),
        make_output(" hello world\n"),
    )
    assert results[0].passed


def test_contains_is_case_insensitive_when_flagged():
    results = score_text(
        make_task([Check(type="contains", value="Paris", flags=["ignore_case"])]),
        make_output("the answer is paris"),
    )
    assert results[0].passed


def test_regex_supports_multiline_output():
    results = score_text(
        make_task([Check(type="regex", value=r"def add\(a, b\):\n\s+return a \+ b")]),
        make_output("def add(a, b):\n    return a + b"),
    )
    assert results[0].passed


def test_json_valid_passes_for_valid_json():
    results = score_text(make_task([Check(type="json_valid")]), make_output('{"answer": 4}'))
    assert results[0].passed


def test_json_field_equals_reads_dot_path():
    results = score_text(
        make_task([Check(type="json_field_equals", path="answer.value", value=4)]),
        make_output('{"answer": {"value": 4}}'),
    )
    assert results[0].passed


def test_json_field_equals_fails_on_missing_path():
    results = score_text(
        make_task([Check(type="json_field_equals", path="answer.value", value=4)]),
        make_output('{"answer": {}}'),
    )
    assert not results[0].passed
    assert "missing path" in results[0].message
