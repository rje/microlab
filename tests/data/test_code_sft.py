from microlab.data.code_sft import (
    is_code_conv,
    normalize_commitpack,
    normalize_mbpp_train,
    oasst_code_convs,
    verified_competitive_rows,
    verify_io,
    verify_unit_test,
)
from microlab.evals.code.tasks import CodeTask


def test_is_code_conv_true_when_assistant_has_fenced_code():
    conv = {"turns": [{"user": "sort a list in python",
                       "assistant": "Use sorted:\n```python\nsorted(xs)\n```"}]}
    assert is_code_conv(conv) is True


def test_is_code_conv_false_for_pure_prose():
    conv = {"turns": [{"user": "hi", "assistant": "Hello, how are you?"}]}
    assert is_code_conv(conv) is False


def test_oasst_code_convs_keeps_only_code_threads():
    # two roots: one code, one prose; only the code one survives
    messages = [
        {"message_id": "a", "parent_id": None, "role": "prompter", "text": "write python",
         "lang": "en"},
        {"message_id": "b", "parent_id": "a", "role": "assistant",
         "text": "```python\nprint(1)\n```", "lang": "en", "rank": 0},
        {"message_id": "c", "parent_id": None, "role": "prompter", "text": "hello", "lang": "en"},
        {"message_id": "d", "parent_id": "c", "role": "assistant", "text": "hi there",
         "lang": "en", "rank": 0},
    ]
    convs = oasst_code_convs(messages)
    assert len(convs) == 1
    assert "print(1)" in convs[0]["turns"][0]["assistant"]


def test_normalize_commitpack_message_to_new_contents():
    row = {"subject": "Fix off-by-one in range", "message": "Fix off-by-one in range\n",
           "new_contents": "for i in range(n):\n    pass\n", "old_contents": "...",
           "lang": "Python"}
    got = normalize_commitpack(row, lang_allow={"python"})
    assert got == {"instruction": "Fix off-by-one in range",
                   "context": "", "response": "for i in range(n):\n    pass"}


def test_normalize_commitpack_drops_disallowed_language():
    row = {"subject": "x", "message": "x", "new_contents": "console.log(1)", "lang": "JavaScript"}
    assert normalize_commitpack(row, lang_allow={"python"}) is None


def test_normalize_commitpack_drops_empty_message_or_body():
    assert normalize_commitpack({"message": "", "new_contents": "x", "lang": "Python"},
                                lang_allow={"python"}) is None
    assert normalize_commitpack({"message": "do", "new_contents": "  ", "lang": "Python"},
                                lang_allow={"python"}) is None


def test_normalize_mbpp_train_prompt_to_code():
    row = {"task_id": 601, "text": "Write a function to add two numbers.",
           "code": "def add(a, b):\n    return a + b", "test_list": ["assert add(1,2)==3"]}
    got = normalize_mbpp_train(row)
    assert got["instruction"] == "Write a function to add two numbers."
    assert got["response"] == "def add(a, b):\n    return a + b"
    assert got["context"] == ""


def test_verify_io_accepts_correct_and_rejects_wrong():
    sol = "n = int(input())\nprint(n * 2)\n"
    assert verify_io(sol, stdin_data="21\n", expected_stdout="42\n") is True
    assert verify_io("print('nope')\n", stdin_data="21\n", expected_stdout="42\n") is False


def test_verify_io_rejects_infinite_loop_via_timeout():
    assert verify_io("while True:\n    pass\n", stdin_data="", expected_stdout="x\n",
                     timeout_s=2.0) is False


def test_verify_unit_test_accepts_correct_solution():
    task = CodeTask(task_id="t", prompt="", instruction="add",
                    entry_point="add",
                    test_program="def check(add):\n    assert add(1, 2) == 3\ncheck(add)\n")
    assert verify_unit_test("def add(a, b):\n    return a + b", task) is True
    assert verify_unit_test("def add(a, b):\n    return a - b", task) is False


def test_verified_competitive_rows_keeps_only_passing_solution():
    problems = [{
        "statement": "Read n, print n*2.",
        "solutions": ["n=int(input());print(n-1)",      # wrong
                      "n=int(input());print(n*2)"],       # correct
        "io": [{"input": "21\n", "output": "42\n"}],
    }]
    rows, tally = verified_competitive_rows(problems, max_per_problem=1)
    assert tally == {"problems": 1, "verified": 1, "no_passing_solution": 0}
    assert len(rows) == 1
    assert rows[0]["instruction"] == "Read n, print n*2."
    assert rows[0]["response"] == "n=int(input());print(n*2)"


def test_verified_competitive_rows_drops_problem_with_no_passing_solution():
    problems = [{"statement": "s", "solutions": ["print('x')"],
                 "io": [{"input": "", "output": "y\n"}]}]
    rows, tally = verified_competitive_rows(problems)
    assert rows == []
    assert tally["no_passing_solution"] == 1
