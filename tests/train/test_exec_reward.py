import pytest

from microlab.train.exec_reward import extract_solution, io_reward, make_exec_score_texts

CASES = [{"input": "21\n", "output": "42\n"}, {"input": "5\n", "output": "10\n"}]


def test_io_reward_fraction_of_cases():
    assert io_reward("n=int(input());print(n*2)", CASES) == 1.0
    # passes only the first case -> 0.5
    assert io_reward("print(42)", CASES) == 0.5
    assert io_reward("print('x')", CASES) == 0.0
    assert io_reward("", CASES) == 0.0          # no code -> 0, not an error


def test_extract_solution_unfences_and_truncates():
    reply = "Here you go:\n```python\nprint(1)\n```\n### End\njunk"
    assert extract_solution(reply) == "print(1)"


def test_score_texts_maps_prompts_and_rejects_unknown():
    score = make_exec_score_texts({"P1": CASES})
    got = score("P1", ["```python\nn=int(input());print(n*2)\n```", "nonsense"])
    assert got[0] == 1.0 and got[1] == 0.0
    with pytest.raises(KeyError):
        score("P-unknown", ["x"])
