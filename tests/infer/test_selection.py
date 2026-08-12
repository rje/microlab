from microlab.infer.selection import (
    behavior_clusters,
    normalize_code,
    pick_from_cluster,
    select_by_self_tests,
    text_plurality,
)


def test_normalize_and_text_plurality():
    cands = ["def f(x):\n    return x+1", "def f(x):\n\treturn x+1",  # same normalized
             "def f(x):\n    return x+2"]
    assert normalize_code(cands[0]) == normalize_code(cands[1])
    assert text_plurality(cands) in (0, 1)          # the duplicated variant wins


def test_behavior_clusters_groups_identical_signatures_largest_first():
    sigs = [("ok", "1"), ("ok", "2"), ("ok", "1"), ("err",), ("ok", "1")]
    clusters = behavior_clusters(sigs)
    assert clusters[0] == [0, 2, 4]                  # largest cluster first
    assert [1] in clusters and [3] in clusters


def test_pick_from_cluster_rules():
    cands = ["longer_candidate_text", "ab", "medium_one"]
    assert pick_from_cluster([0, 1, 2], cands, rule="shortest") == 1
    r = pick_from_cluster([0, 1, 2], cands, rule="random", seed=7)
    assert r in (0, 1, 2)
    assert pick_from_cluster([0, 1, 2], cands, rule="random", seed=7) == r  # deterministic


def test_select_by_self_tests_argmax_first_tie():
    assert select_by_self_tests([1, 3, 3, 0]) == 1
