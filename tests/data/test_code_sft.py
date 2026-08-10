from microlab.data.code_sft import (
    benchmark_fingerprints,
    decontaminate,
    is_code_conv,
    normalize_commitpack,
    normalize_mbpp_train,
    oasst_code_convs,
    row_supervised_tokens,
    token_match_subsample,
    total_supervised_tokens,
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


def test_verify_io_exit_call_cannot_bypass_comparison():
    # a wrong solution that self-terminates must NOT pass
    assert verify_io("print('WRONG'); exit()", stdin_data="21\n", expected_stdout="42\n") is False
    assert verify_io("import sys; print('WRONG'); sys.exit(0)", stdin_data="21\n",
                     expected_stdout="42\n") is False


def test_verify_io_correct_solution_with_exit_still_passes():
    # a CORRECT solution that happens to call exit() must still pass
    assert verify_io("n=int(input()); print(n*2); exit()", stdin_data="21\n",
                     expected_stdout="42\n") is True


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


def test_verified_competitive_rows_caps_cases_and_solutions():
    # max_cases limits verification to the leading cases: a solution that passes the first
    # case but would fail a later one is accepted when max_cases=1 (the sanity-filter tradeoff
    # for already-accepted human solutions).
    problems = [{
        "statement": "double n",
        "solutions": ["n=int(input());print(n*2)"],
        "io": [{"input": "21\n", "output": "42\n"},   # solution passes this
               {"input": "5\n", "output": "999\n"}],   # ...and fails this (bogus expected)
    }]
    rows, tally = verified_competitive_rows(problems, max_cases=1)
    assert tally["verified"] == 1 and len(rows) == 1  # only the first case was checked
    # without the cap, the bad second case drops the problem
    rows2, tally2 = verified_competitive_rows(problems)
    assert tally2["no_passing_solution"] == 1 and rows2 == []


def test_apps_problem_coerces_nonstring_io_and_skips_call_based():
    import json as _j

    from microlab.data.code_sft import apps_problem
    # standard stdin/stdout problem with a stray non-string output item (observed in TACO):
    # a bool inside the outputs list must not crash — it is str()'d.
    row = {"question": "Q", "solutions": _j.dumps(["print(1)"]),
           "input_output": _j.dumps({"inputs": ["1\n"], "outputs": [[True]]})}
    p = apps_problem(row)
    assert p["io"] == [{"input": "1\n", "output": "True"}]
    # call-based problems (fn_name present) are stdin/stdout-incompatible -> no cases
    call_based = {"question": "Q", "solutions": _j.dumps(["def f(): pass"]),
                  "input_output": _j.dumps({"fn_name": "f", "inputs": [[1]], "outputs": [[2]]})}
    assert apps_problem(call_based)["io"] == []


def test_apps_problem_survives_huge_integer_test_case():
    import json as _j

    from microlab.data.code_sft import apps_problem
    # a competitive test case with a 9000-digit integer exceeds Python 3.11's int()-from-str
    # digit cap; the adapter must parse it as a string, not crash (json here writes it as a
    # bare number so json.loads would otherwise int() it).
    big = "9" * 9000
    io = f'{{"inputs": [{big}], "outputs": ["ok\\n"]}}'
    row = {"question": "Q", "solutions": _j.dumps(["print('ok')"]), "input_output": io}
    p = apps_problem(row)  # must not raise
    assert p["io"] == [{"input": big, "output": "ok\n"}]


def test_taco_problem_shares_io_normalization():
    import json as _j

    from microlab.data.code_sft import taco_problem
    row = {"question": "Q", "solutions": _j.dumps(["print(2)"]),
           "input_output": _j.dumps({"inputs": [["a", "b"]], "outputs": ["ab\n"]})}
    # a list-of-lines input is joined; output passes through
    assert taco_problem(row)["io"] == [{"input": "ab", "output": "ab\n"}]


class _ByteTok:
    def encode(self, s): return list(s.encode("utf-8"))


def test_row_supervised_tokens_counts_response_plus_sentinel():
    from microlab.model.reference.chat_sft import END_SENTINEL
    tok = _ByteTok()
    row = {"instruction": "hi", "context": "", "response": "print(1)"}
    assert row_supervised_tokens(row, tok) == len(tok.encode("print(1)" + END_SENTINEL))


def test_token_match_subsample_hits_target_within_one_row():
    tok = _ByteTok()
    rows = [{"instruction": "i", "context": "", "response": "x" * 10} for _ in range(100)]
    per = row_supervised_tokens(rows[0], tok)
    target = per * 12
    got = token_match_subsample(rows, target_tokens=target, tok=tok, seed=0)
    assert abs(total_supervised_tokens(got, tok) - target) <= per  # within one row


def test_decontaminate_removes_planted_benchmark_row_keeps_benign():
    bench_text = "def has_close_elements(numbers, threshold): return any(abs(a-b) < threshold)"
    bench = [bench_text]
    fp = benchmark_fingerprints(bench, n=8)
    rows = [
        {"instruction": "impl", "context": "", "response": bench_text},
        {"instruction": "add", "context": "", "response": "def add(a, b):\n    return a + b"},
    ]
    kept, removed = decontaminate(rows, fp, n=8)
    assert removed == 1
    assert kept == [rows[1]]


def test_decontaminate_keeps_uncontaminated_rows():
    bench = ["def foo(x): return x + 1"]
    fp = benchmark_fingerprints(bench, n=5)
    rows = [
        {"instruction": "impl bar", "context": "", "response": "def bar(y): return y * 2"},
        {"instruction": "sort list", "context": "", "response": "return sorted(arr)"},
    ]
    kept, removed = decontaminate(rows, fp, n=5)
    assert removed == 0
    assert kept == rows


def test_decontaminate_handles_multiturn_rows():
    bench = ["def fibonacci(n): return n"]
    fp = benchmark_fingerprints(bench, n=5)
    rows = [
        {"turns": [
            {"user": "write fibonacci", "assistant": "def fibonacci(n): return n"},
            {"user": "test it", "assistant": "print(fibonacci(5))"},
        ]},
        {"turns": [
            {"user": "write sum", "assistant": "def sum_nums(arr): return sum(arr)"},
        ]},
    ]
    kept, removed = decontaminate(rows, fp, n=5)
    assert removed == 1  # first row matches benchmark
    assert kept == [rows[1]]


def test_decontaminate_empty_fingerprints_keeps_all():
    rows = [
        {"instruction": "a", "context": "", "response": "def x(): pass"},
        {"instruction": "b", "context": "", "response": "def y(): pass"},
    ]
    kept, removed = decontaminate(rows, set(), n=5)
    assert removed == 0
    assert kept == rows


def test_decontaminate_case_insensitive_normalization():
    # Test that tokens are normalized to lowercase for matching
    bench = ["def hello"]
    fp = benchmark_fingerprints(bench, n=2)
    rows = [
        {"instruction": "", "context": "", "response": "DEF HELLO"},
    ]
    kept, removed = decontaminate(rows, fp, n=2)
    assert removed == 1  # should match case-insensitively
