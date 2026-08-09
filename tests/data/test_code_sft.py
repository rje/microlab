from microlab.data.code_sft import normalize_commitpack, normalize_mbpp_train


def test_normalize_commitpack_message_to_new_contents():
    row = {"subject": "Fix off-by-one in range", "message": "Fix off-by-one in range\n",
           "new_contents": "for i in range(n):\n    pass\n", "old_contents": "...",
           "lang": "Python"}
    got = normalize_commitpack(row, lang_allow={"python"})
    assert got == {"instruction": "Fix off-by-one in range",
                   "context": "", "response": "for i in range(n):\n    pass\n"}


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
