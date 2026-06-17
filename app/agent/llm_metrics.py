from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

MAX_RECENT_CALLS = 200


@dataclass
class LLMCallRecord:
    operation: str  # "plan" | "repair" | "synthesize"
    ok: bool
    latency_ms: float
    total_tokens: int | None = None
    error: str | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class RepairRecord:
    n_problems_in: int
    n_fixed: int
    n_dropped: int
    timestamp: float = field(default_factory=time.time)


class LLMMetrics:
    """In-process, fixed-window LLM call metrics.

    Deliberately not a time-series/Prometheus setup - this is a single-process
    local-dev tool, so a bounded in-memory ring buffer is enough to answer
    "is the LLM slow/flaky right now" without adding an external dependency.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[LLMCallRecord] = []
        self._repairs: list[RepairRecord] = []

    def record(self, rec: LLMCallRecord) -> None:
        with self._lock:
            self._records.append(rec)
            if len(self._records) > MAX_RECENT_CALLS:
                self._records = self._records[-MAX_RECENT_CALLS:]

    def record_repair(self, rec: RepairRecord) -> None:
        with self._lock:
            self._repairs.append(rec)
            if len(self._repairs) > MAX_RECENT_CALLS:
                self._repairs = self._repairs[-MAX_RECENT_CALLS:]

    def repair_snapshot(self) -> dict:
        with self._lock:
            repairs = list(self._repairs)

        total = len(repairs)
        if not total:
            return {
                "repair_attempts": 0,
                "total_problems": 0,
                "total_fixed": 0,
                "total_dropped": 0,
                "fix_rate": 0.0,
            }

        total_problems = sum(r.n_problems_in for r in repairs)
        total_fixed = sum(r.n_fixed for r in repairs)
        total_dropped = sum(r.n_dropped for r in repairs)

        return {
            "repair_attempts": total,
            "total_problems": total_problems,
            "total_fixed": total_fixed,
            "total_dropped": total_dropped,
            "fix_rate": round(total_fixed / total_problems, 4) if total_problems else 0.0,
        }

    def snapshot(self) -> dict:
        with self._lock:
            records = list(self._records)

        total = len(records)
        errors = sum(1 for r in records if not r.ok)
        avg_latency = sum(r.latency_ms for r in records) / total if total else 0.0
        total_tokens = sum(r.total_tokens or 0 for r in records)

        by_operation: dict[str, dict] = {}
        for r in records:
            bucket = by_operation.setdefault(r.operation, {"count": 0, "errors": 0, "_total_latency_ms": 0.0})
            bucket["count"] += 1
            bucket["errors"] += 0 if r.ok else 1
            bucket["_total_latency_ms"] += r.latency_ms
        for bucket in by_operation.values():
            bucket["avg_latency_ms"] = round(bucket.pop("_total_latency_ms") / bucket["count"], 2)

        return {
            "window_size": total,
            "error_count": errors,
            "error_rate": round(errors / total, 4) if total else 0.0,
            "avg_latency_ms": round(avg_latency, 2),
            "total_tokens_sampled": total_tokens,
            "by_operation": by_operation,
        }


metrics = LLMMetrics()
