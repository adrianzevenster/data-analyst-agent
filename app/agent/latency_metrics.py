"""In-process ring buffer for per-turn latency measurements."""
from __future__ import annotations

import threading
from collections import deque

_WINDOW = 200


class LatencyMetrics:
    """Collects t_plan / t_execute / t_synthesis (ms) for the last N turns."""

    def __init__(self, window: int = _WINDOW) -> None:
        self._lock = threading.Lock()
        self._records: deque[dict[str, float]] = deque(maxlen=window)

    def record(self, t_plan: float, t_execute: float, t_synthesis: float) -> None:
        with self._lock:
            self._records.append({
                "t_plan": round(t_plan, 1),
                "t_execute": round(t_execute, 1),
                "t_synthesis": round(t_synthesis, 1),
                "t_total": round(t_plan + t_execute + t_synthesis, 1),
            })

    def snapshot(self) -> dict:
        with self._lock:
            recs = list(self._records)
        if not recs:
            return {"n_turns": 0, "phases": {}}

        phases = ["t_plan", "t_execute", "t_synthesis", "t_total"]
        stats: dict[str, dict] = {}
        for phase in phases:
            vals = sorted(r[phase] for r in recs)
            n = len(vals)
            avg = sum(vals) / n
            p50 = vals[int(n * 0.50)]
            p95 = vals[min(int(n * 0.95), n - 1)]
            stats[phase] = {"avg_ms": round(avg, 1), "p50_ms": round(p50, 1), "p95_ms": round(p95, 1)}

        return {"n_turns": len(recs), "phases": stats}


latency_metrics = LatencyMetrics()
