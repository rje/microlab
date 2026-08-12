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
