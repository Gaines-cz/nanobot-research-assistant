"""RAG Evaluation - Metrics calculator."""

from typing import List, Optional

from nanobot.rag.evaluation.base import EvalResult


class MetricsCalculator:
    """Metrics calculator for RAG evaluation."""

    @staticmethod
    def recall_at_k(results: List[EvalResult], k: int = 5, use_baseline: bool = False) -> float:
        """Calculate Recall@K."""
        if not results:
            return 0.0

        hits = 0
        for r in results:
            if use_baseline:
                if r.baseline_hit and r.baseline_hit_rank and r.baseline_hit_rank <= k:
                    hits += 1
            else:
                if r.hit and r.hit_rank and r.hit_rank <= k:
                    hits += 1

        return hits / len(results)

    @staticmethod
    def mrr(results: List[EvalResult], use_baseline: bool = False) -> float:
        """Calculate Mean Reciprocal Rank."""
        if not results:
            return 0.0

        total = 0.0

        for r in results:
            if use_baseline:
                if r.baseline_hit and r.baseline_hit_rank:
                    total += 1.0 / r.baseline_hit_rank
            else:
                if r.hit and r.hit_rank:
                    total += 1.0 / r.hit_rank

        return total / len(results)

    @staticmethod
    def hit_rate_at_k(results: List[EvalResult], k: int = 5, use_baseline: bool = False) -> float:
        """Calculate Hit Rate@K (whether any hit in top-K)."""
        if not results:
            return 0.0

        hits = 0
        for r in results:
            if use_baseline:
                if r.baseline_hit and r.baseline_hit_rank and r.baseline_hit_rank <= k:
                    hits += 1
            else:
                if r.hit and r.hit_rank and r.hit_rank <= k:
                    hits += 1

        return hits / len(results)

    @staticmethod
    def avg_latency(results: List[EvalResult]) -> float:
        """Calculate average latency."""
        if not results:
            return 0.0
        return sum(r.latency_ms for r in results) / len(results)

    @staticmethod
    def difficulty_breakdown(results: List[EvalResult]) -> dict:
        """Break down metrics by difficulty."""
        if not results:
            return {}

        breakdown = {}
        for r in results:
            if r.difficulty:
                if r.difficulty not in breakdown:
                    breakdown[r.difficulty] = {"total": 0, "hits": 0}
                breakdown[r.difficulty]["total"] += 1
                if r.hit:
                    breakdown[r.difficulty]["hits"] += 1

        # Calculate recall for each difficulty
        for diff in breakdown:
            total = breakdown[diff]["total"]
            hits = breakdown[diff]["hits"]
            breakdown[diff]["recall"] = hits / total if total > 0 else 0.0

        return breakdown

    @staticmethod
    def failure_breakdown(results: List[EvalResult]) -> dict:
        """Break down by failure reason."""
        if not results:
            return {}

        breakdown = {}
        for r in results:
            if not r.hit and r.failure_reason:
                reason = r.failure_reason
                if reason not in breakdown:
                    breakdown[reason] = 0
                breakdown[reason] += 1

        return breakdown

    @staticmethod
    def compute_relative_improvement(
        our_metric: float,
        baseline_metric: float,
    ) -> Optional[float]:
        """Calculate relative improvement (%)."""
        if baseline_metric == 0:
            return None
        return (our_metric - baseline_metric) / baseline_metric * 100
