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


_MAX_TRACKED_MODELS = 50


class ScoringLatencyTracker:
    """Per-model-id ring buffer of scoring latency measurements."""

    def __init__(self, window: int = 200) -> None:
        self._lock = threading.Lock()
        self._by_model: dict[str, deque] = {}
        self._window = window

    def record(self, model_id: str, latency_ms: float) -> None:
        with self._lock:
            if model_id not in self._by_model:
                if len(self._by_model) >= _MAX_TRACKED_MODELS:
                    # Evict the model with the oldest last entry.
                    oldest = min(self._by_model, key=lambda k: self._by_model[k][-1] if self._by_model[k] else 0)
                    del self._by_model[oldest]
                self._by_model[model_id] = deque(maxlen=self._window)
            self._by_model[model_id].append(round(latency_ms, 1))

    def snapshot(self) -> dict:
        with self._lock:
            copy = {mid: list(vals) for mid, vals in self._by_model.items()}
        result: dict[str, dict] = {}
        for mid, vals in copy.items():
            if not vals:
                continue
            sorted_vals = sorted(vals)
            n = len(sorted_vals)
            result[mid] = {
                "n": n,
                "avg_ms": round(sum(sorted_vals) / n, 1),
                "p50_ms": sorted_vals[int(n * 0.50)],
                "p95_ms": sorted_vals[min(int(n * 0.95), n - 1)],
            }
        return {"n_models": len(result), "by_model": result}


scoring_latency = ScoringLatencyTracker()
