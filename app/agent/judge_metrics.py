from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

MAX_RECENT_JUDGEMENTS = 200
LOW_SCORE_THRESHOLD = 3


@dataclass
class JudgeRecord:
    score: int
    issue_count: int
    timestamp: float = field(default_factory=time.time)


class JudgeMetrics:
    """Aggregates sampled LLM-as-judge groundedness scores.

    Separate from LLMMetrics: that tracks call performance (latency/tokens/
    errors) for every LLM call including judge calls themselves; this tracks
    the judge's actual verdicts (is the synthesized answer grounded), which
    is a quality signal rather than a performance one.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[JudgeRecord] = []

    def record(self, rec: JudgeRecord) -> None:
        with self._lock:
            self._records.append(rec)
            if len(self._records) > MAX_RECENT_JUDGEMENTS:
                self._records = self._records[-MAX_RECENT_JUDGEMENTS:]

    def snapshot(self) -> dict:
        with self._lock:
            records = list(self._records)

        total = len(records)
        avg_score = sum(r.score for r in records) / total if total else 0.0
        low_score_count = sum(1 for r in records if r.score <= LOW_SCORE_THRESHOLD)
        flagged_count = sum(1 for r in records if r.issue_count > 0)

        return {
            "sampled_count": total,
            "avg_groundedness_score": round(avg_score, 2),
            "low_score_rate": round(low_score_count / total, 4) if total else 0.0,
            "flagged_rate": round(flagged_count / total, 4) if total else 0.0,
        }


judge_metrics = JudgeMetrics()
