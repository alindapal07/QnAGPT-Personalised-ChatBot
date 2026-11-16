"""In-memory telemetry and observability metrics for CRAG."""

from dataclasses import dataclass, field
import threading
import time
from typing import Any, Dict, List

from qna_bot.rag.crag.models import CRAGEvaluation, RelevanceGrade


@dataclass
class CRAGTelemetrySnapshot:
    """Read-only view of aggregated CRAG metrics."""

    total_evaluations: int = 0
    correct_count: int = 0
    ambiguous_count: int = 0
    incorrect_count: int = 0
    contradictory_count: int = 0
    rewrite_count: int = 0
    web_fallback_count: int = 0
    abstention_count: int = 0
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0


class CRAGMetricsTracker:
    """Thread-safe telemetry tracker recording CRAG decisions and latencies."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total_evaluations = 0
        self._grade_counts: Dict[str, int] = {
            RelevanceGrade.CORRECT.value: 0,
            RelevanceGrade.AMBIGUOUS.value: 0,
            RelevanceGrade.INCORRECT.value: 0,
            RelevanceGrade.CONTRADICTORY.value: 0,
        }
        self._rewrite_count = 0
        self._web_fallback_count = 0
        self._abstention_count = 0
        self._latencies_ms: List[float] = []

    def record_evaluation(self, evaluation: CRAGEvaluation, latency_ms: float = 0.0) -> None:
        """Record a completed document evaluation."""
        with self._lock:
            self._total_evaluations += 1
            grade_key = evaluation.relevance_grade.value
            self._grade_counts[grade_key] = self._grade_counts.get(grade_key, 0) + 1
            if latency_ms > 0:
                self._latencies_ms.append(latency_ms)
                if len(self._latencies_ms) > 1000:
                    self._latencies_ms.pop(0)

    def record_rewrite(self) -> None:
        """Increment count of query rewrites."""
        with self._lock:
            self._rewrite_count += 1

    def record_web_fallback(self) -> None:
        """Increment count of web search fallbacks."""
        with self._lock:
            self._web_fallback_count += 1

    def record_abstention(self) -> None:
        """Increment count of honest abstentions."""
        with self._lock:
            self._abstention_count += 1

    def get_summary(self) -> Dict[str, Any]:
        """Return a structured dictionary of metrics."""
        with self._lock:
            tot = self._total_evaluations
            avg_lat = sum(self._latencies_ms) / max(1, len(self._latencies_ms)) if self._latencies_ms else 0.0
            return {
                "total_evaluations": tot,
                "grade_distribution": dict(self._grade_counts),
                "query_rewrites": self._rewrite_count,
                "web_fallbacks": self._web_fallback_count,
                "abstentions": self._abstention_count,
                "avg_evaluation_latency_ms": round(avg_lat, 2),
            }

    def reset(self) -> None:
        """Reset all counters (primarily for unit tests)."""
        with self._lock:
            self._total_evaluations = 0
            for k in self._grade_counts:
                self._grade_counts[k] = 0
            self._rewrite_count = 0
            self._web_fallback_count = 0
            self._abstention_count = 0
            self._latencies_ms.clear()


# Global default tracker
_DEFAULT_TRACKER = CRAGMetricsTracker()


def get_crag_metrics() -> CRAGMetricsTracker:
    """Return the global default CRAG metrics tracker."""
    return _DEFAULT_TRACKER
