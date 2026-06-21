from __future__ import annotations

from app.agent.judge_metrics import JudgeMetrics, JudgeRecord


def test_snapshot_empty():
    m = JudgeMetrics(db_path=":memory:")

    snap = m.snapshot()

    assert snap == {
        "response_count": 0,
        "eligible_count": 0,
        "attempted_count": 0,
        "sampled_count": 0,
        "skipped_count": 0,
        "skipped_sample_rate_count": 0,
        "skipped_rule_based_count": 0,
        "skipped_llm_disabled_count": 0,
        "error_count": 0,
        "avg_groundedness_score": 0.0,
        "low_score_rate": 0.0,
        "flagged_rate": 0.0,
        "last_error": None,
    }


def test_snapshot_aggregates_scores_and_flags():
    m = JudgeMetrics(db_path=":memory:")
    m.record(JudgeRecord(score=5, issue_count=0))
    m.record(JudgeRecord(score=2, issue_count=1))
    m.record(JudgeRecord(score=3, issue_count=0))

    snap = m.snapshot()

    assert snap["sampled_count"] == 3
    assert snap["response_count"] == 3
    assert snap["eligible_count"] == 3
    assert snap["attempted_count"] == 3
    assert snap["avg_groundedness_score"] == round((5 + 2 + 3) / 3, 2)
    assert snap["low_score_rate"] == round(2 / 3, 4)  # scores 2 and 3 are <= threshold of 3
    assert snap["flagged_rate"] == round(1 / 3, 4)


def test_snapshot_tracks_skips_and_failures():
    m = JudgeMetrics(db_path=":memory:")
    m.record_skipped("not_sampled")
    m.record_skipped("rule_based")
    m.record_skipped("llm_disabled")
    m.record_failure("invalid JSON")

    snap = m.snapshot()

    assert snap["response_count"] == 4
    assert snap["eligible_count"] == 2
    assert snap["attempted_count"] == 1
    assert snap["sampled_count"] == 0
    assert snap["skipped_count"] == 3
    assert snap["skipped_sample_rate_count"] == 1
    assert snap["skipped_rule_based_count"] == 1
    assert snap["skipped_llm_disabled_count"] == 1
    assert snap["error_count"] == 1
    assert snap["last_error"] == "invalid JSON"
