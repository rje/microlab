from microlab.data.reference.loaders import load_sample
from microlab.data.reference.pipeline import (
    clean_text,
    dedup_docs,
    filter_contamination,
    split_docs,
)


def test_clean_collapses_spaces_strips_control_preserves_tabs():
    # 3 spaces collapse to 1, the NUL control char is removed, the tab is preserved,
    # and the trailing space is rstripped per line.
    assert clean_text("a   b\x00c\t d ") == "a bc\t d"


def test_dedup_preserves_order_and_removes_dupes():
    assert dedup_docs(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_contamination_drops_matching_docs():
    docs = ["clean doc", "leaked: What is 2+2?", "fine"]
    assert filter_contamination(docs, ["What is 2+2?"]) == ["clean doc", "fine"]


def test_split_is_disjoint_deterministic_and_covers_all():
    docs = [f"doc-{i}" for i in range(100)]
    s1 = split_docs(docs, seed=0)
    s2 = split_docs(docs, seed=0)
    assert s1 == s2  # deterministic
    allsplit = s1["train"] + s1["val"] + s1["test"]
    assert sorted(allsplit) == sorted(docs)  # covers all, disjoint
    assert len(s1["train"]) == 80 and len(s1["val"]) == 10 and len(s1["test"]) == 10


def test_sample_loads_and_is_nonempty():
    assert len(load_sample()) > 200
