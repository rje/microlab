import pytest

from microlab.evals.schema import Check, EvalTask


def test_eval_task_accepts_valid_payload():
    task = EvalTask.from_dict(
        {
            "id": "exact-001",
            "category": "math",
            "prompt": "Answer only with 4.",
            "checks": [{"type": "exact_match", "value": "4"}],
            "max_new_tokens": 8,
            "metadata": {"difficulty": "smoke"},
        }
    )

    assert task.id == "exact-001"
    assert task.checks == [Check(type="exact_match", value="4")]
    assert task.max_new_tokens == 8


def test_eval_task_rejects_missing_checks():
    with pytest.raises(ValueError, match="at least one check"):
        EvalTask.from_dict(
            {
                "id": "bad-001",
                "category": "math",
                "prompt": "Answer only with 4.",
                "checks": [],
            }
        )


def test_eval_task_rejects_unknown_check_type():
    with pytest.raises(ValueError, match="unsupported check type"):
        EvalTask.from_dict(
            {
                "id": "bad-002",
                "category": "math",
                "prompt": "Answer only with 4.",
                "checks": [{"type": "semantic_vibes", "value": "good"}],
            }
        )
