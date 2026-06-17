from __future__ import annotations

from app.agent.llm_metrics import MAX_RECENT_CALLS, LLMCallRecord, LLMMetrics


def test_snapshot_empty_metrics():
    m = LLMMetrics()

    snap = m.snapshot()

    assert snap == {
        "window_size": 0,
        "error_count": 0,
        "error_rate": 0.0,
        "avg_latency_ms": 0.0,
        "total_tokens_sampled": 0,
        "by_operation": {},
    }


def test_snapshot_aggregates_by_operation():
    m = LLMMetrics()
    m.record(LLMCallRecord(operation="plan", ok=True, latency_ms=100.0, total_tokens=50))
    m.record(LLMCallRecord(operation="plan", ok=False, latency_ms=200.0, error="boom"))
    m.record(LLMCallRecord(operation="synthesize", ok=True, latency_ms=300.0, total_tokens=70))

    snap = m.snapshot()

    assert snap["window_size"] == 3
    assert snap["error_count"] == 1
    assert snap["error_rate"] == round(1 / 3, 4)
    assert snap["total_tokens_sampled"] == 120
    assert snap["by_operation"]["plan"]["count"] == 2
    assert snap["by_operation"]["plan"]["errors"] == 1
    assert snap["by_operation"]["plan"]["avg_latency_ms"] == 150.0
    assert snap["by_operation"]["synthesize"]["count"] == 1
    assert snap["by_operation"]["synthesize"]["errors"] == 0


def test_ring_buffer_caps_at_max_recent_calls():
    m = LLMMetrics()
    for i in range(MAX_RECENT_CALLS + 10):
        m.record(LLMCallRecord(operation="plan", ok=True, latency_ms=1.0))

    snap = m.snapshot()

    assert snap["window_size"] == MAX_RECENT_CALLS
