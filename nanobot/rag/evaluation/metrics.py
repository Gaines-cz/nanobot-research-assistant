"""RAG Evaluation - Metrics calculator."""

import math
from typing import List

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
    def ndcg_at_k(results: List[EvalResult], k: int = 5, use_baseline: bool = False) -> float:
        """Calculate NDCG@K (Normalized Discounted Cumulative Gain).

        For binary relevance (hit=1, miss=0):
        DCG@k = sum(rel_i / log2(i+1)) for i in 1 to k
        IDCG@k = DCG@k for perfect ranking (all hits at top)
        NDCG@k = DCG@k / IDCG@k
        """
        if not results:
            return 0.0

        # Build relevance list for this result
        def calc_dcg(relevances: List[int], k: int) -> float:
            dcg = 0.0
            for i, rel in enumerate(relevances[:k]):
                dcg += rel / math.log2(i + 2)  # i+2 because i is 0-indexed, log2(1)=0
            return dcg

        total_ndcg = 0.0
        for r in results:
            if use_baseline:
                if r.baseline_hit and r.baseline_hit_rank:
                    # Relevance: 1 for the hit at its rank, 0 elsewhere
                    relevances = [1 if i + 1 == r.baseline_hit_rank else 0 for i in range(k)]
                else:
                    relevances = [0] * k
            else:
                if r.hit and r.hit_rank:
                    relevances = [1 if i + 1 == r.hit_rank else 0 for i in range(k)]
                else:
                    relevances = [0] * k

            dcg = calc_dcg(relevances, k)
            # IDCG: best possible DCG (hit at position 1)
            idcg = calc_dcg([1], k)
            ndcg = dcg / idcg if idcg > 0 else 0.0
            total_ndcg += ndcg

        return total_ndcg / len(results)

    @staticmethod
    def avg_latency(results: List[EvalResult]) -> float:
        """Calculate average latency in ms."""
        if not results:
            return 0.0
        return sum(r.latency_ms for r in results) / len(results)

    @staticmethod
    def question_type_breakdown(results: List[EvalResult], queries_map: dict) -> dict:
        """Break down metrics by question type."""
        if not results:
            return {}

        breakdown = {}
        for r in results:
            query = queries_map.get(r.query_id)
            if not query or not query.question_type:
                continue

            qtype = query.question_type
            if qtype not in breakdown:
                breakdown[qtype] = {"total": 0, "hits": 0, "mrr_sum": 0.0}

            breakdown[qtype]["total"] += 1
            if r.hit:
                breakdown[qtype]["hits"] += 1
            if r.hit and r.hit_rank:
                breakdown[qtype]["mrr_sum"] += 1.0 / r.hit_rank

        # Calculate metrics for each type
        for qtype in breakdown:
            total = breakdown[qtype]["total"]
            hits = breakdown[qtype]["hits"]
            breakdown[qtype]["recall"] = hits / total if total > 0 else 0.0
            breakdown[qtype]["mrr"] = breakdown[qtype]["mrr_sum"] / total if total > 0 else 0.0

        return breakdown
