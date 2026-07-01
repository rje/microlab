import pytest

from microlab.evals.story_eval import RUBRIC, parse_scores, run_story_eval


def test_parse_scores_valid_and_clamped():
    s = parse_scores('{"grammar":4,"coherence":3,"creativity":9,"consistency":0}')
    assert s["grammar"] == 4 and s["creativity"] == 5 and s["consistency"] == 1  # clamped 1..5


def test_parse_scores_malformed_defaults_to_three():
    s = parse_scores("not json")
    assert all(s[r] == 3 for r in RUBRIC)


def test_run_story_eval_aggregates_with_mock_judge():
    comps = [{"prompt": "a", "completion": "x"}, {"prompt": "b", "completion": "y"}]
    def judge(p, c):
        return {"grammar": 4, "coherence": 2, "creativity": 3, "consistency": 5}
    res = run_story_eval(comps, judge)
    assert res["means"]["grammar"] == pytest.approx(4.0)
    assert res["means"]["overall"] == pytest.approx((4 + 2 + 3 + 5) / 4)
    assert len(res["samples"]) == 2 and res["samples"][0]["scores"]["grammar"] == 4


def test_run_story_eval_empty():
    assert run_story_eval([], lambda p, c: {}) == {"means": {}, "samples": []}
