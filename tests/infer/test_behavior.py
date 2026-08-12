from microlab.infer.behavior import behavior_signature, signatures_for

ADD = "def add(a, b):\n    return a + b"
BAD = "def add(a, b):\n    return a - b"
BOOM = "def add(a, b):\n    raise ValueError"


def test_signature_distinguishes_right_from_wrong():
    s_good = behavior_signature(ADD, "add", "add(2, 3)")
    s_bad = behavior_signature(BAD, "add", "add(2, 3)")
    assert s_good[0] == "ok" and s_bad[0] == "ok" and s_good != s_bad


def test_signature_error_and_vector():
    assert behavior_signature(BOOM, "add", "add(1, 1)") == ("err",)
    vec = signatures_for(ADD, "add", ["add(1, 1)", "add(2, 2)"])
    assert vec == (("ok", "2"), ("ok", "4"))


def test_signature_brace_heavy_candidate_is_safe():
    # str.format must not reprocess braces inside the substituted candidate; this guards
    # against future template edits adding literal braces (which WOULD require {{}} escaping).
    cand = "def f(x):\n    d = {'a': x, 'b': {1, 2}}\n    return f'{d[\"a\"]}'"
    assert behavior_signature(cand, "f", "f(7)") == ("ok", "'7'")
