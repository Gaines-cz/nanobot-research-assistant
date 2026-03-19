"""RAG Evaluation - Report Generator."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from nanobot.rag.evaluation.ablation import AblationConfig
from nanobot.rag.evaluation.base import EvalQuery, EvalResult, EvalSummary


class FailureCategory(Enum):
    """Categories for classifying retrieval failures."""
    CHUNKING_ISSUE = "chunking_issue"
    RETRIEVAL_BM25_FAIL = "bm25_fail"
    RETRIEVAL_VECTOR_FAIL = "vector_fail"
    RERANKING_FAIL = "reranking_fail"
    THRESHOLD_FILTERED = "threshold_filtered"
    SEMANTIC_MISMATCH = "semantic_mismatch"
    UNKNOWN = "unknown"


@dataclass
class FailureAnalysis:
    """Result of failure analysis for a single query."""
    category: FailureCategory
    description: str
    suggested_fix: Optional[str] = None


class FailureAnalyzer:
    """Analyzer for classification retrieval failures."""

    def analyze_failure(
        self,
        result: EvalResult,
        query: EvalQuery,
    ) -> FailureAnalysis:
        """Classify the reason for a retrieval failure."""
        if result.hit:
            return FailureAnalysis(
                category=FailureCategory.UNKNOWN,
                description="Query succeeded, no failure to analyze"
            )

        failure_reason = result.failure_reason or ""

        if "no_results" in failure_reason:
            return FailureAnalysis(
                category=FailureCategory.RETRIEVAL_BM25_FAIL
                if "bm25" in failure_reason.lower()
                else FailureCategory.RETRIEVAL_VECTOR_FAIL,
                description=f"No results retrieved: {failure_reason}",
                suggested_fix="Consider lowering retrieval thresholds or expanding recall_top_k"
            )

        if "not_retrieved" in failure_reason:
            return FailureAnalysis(
                category=FailureCategory.RETRIEVAL_BM25_FAIL,
                description=f"Source chunk not in initial retrieval: {failure_reason}",
                suggested_fix="Check if chunk was indexed, consider larger chunk size or lower thresholds"
            )

        if "semantic" in failure_reason.lower() or "similarity" in failure_reason.lower():
            return FailureAnalysis(
                category=FailureCategory.SEMANTIC_MISMATCH,
                description=f"Semantic similarity too low: {failure_reason}",
                suggested_fix="Consider using a different embedding model or fine-tuning"
            )

        if "threshold" in failure_reason.lower():
            return FailureAnalysis(
                category=FailureCategory.THRESHOLD_FILTERED,
                description=f"Result filtered by threshold: {failure_reason}",
                suggested_fix="Lower the relevant threshold in config"
            )

        return FailureAnalysis(
            category=FailureCategory.UNKNOWN,
            description=f"Unclassified failure: {failure_reason}",
            suggested_fix="Manual investigation recommended"
        )

    def generate_improvement_suggestions(
        self,
        failures: List[FailureAnalysis],
    ) -> List[str]:
        """Generate improvement suggestions based on failure distribution."""
        if not failures:
            return ["No failures to analyze!"]

        category_counts: Dict[FailureCategory, int] = {}
        for failure in failures:
            category_counts[failure.category] = category_counts.get(failure.category, 0) + 1

        suggestions = []
        total = len(failures)

        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total) * 100
            if percentage < 5:
                continue

            if category == FailureCategory.CHUNKING_ISSUE:
                suggestions.append(
                    f"Chunking issues ({percentage:.1f}%): Consider adjusting chunk size"
                )
            elif category == FailureCategory.RETRIEVAL_BM25_FAIL:
                suggestions.append(
                    f"BM25 retrieval failures ({percentage:.1f}%): Try lowering bm25_threshold"
                )
            elif category == FailureCategory.RETRIEVAL_VECTOR_FAIL:
                suggestions.append(
                    f"Vector retrieval failures ({percentage:.1f}%): Try lowering vector_threshold"
                )
            elif category == FailureCategory.RERANKING_FAIL:
                suggestions.append(
                    f"Reranking failures ({percentage:.1f}%): Check rerank_threshold"
                )
            elif category == FailureCategory.THRESHOLD_FILTERED:
                suggestions.append(
                    f"Threshold filtered ({percentage:.1f}%): Consider relaxing thresholds"
                )
            elif category == FailureCategory.SEMANTIC_MISMATCH:
                suggestions.append(
                    f"Semantic mismatch ({percentage:.1f}%): Consider query expansion"
                )

        if not suggestions:
            suggestions.append("No clear patterns identified - manual review recommended")

        return suggestions


class ReportGenerator:
    """Generator for evaluation reports."""

    def __init__(self):
        self.failure_analyzer = FailureAnalyzer()

    @staticmethod
    def generate_ablation_table(
        results: Dict[str, EvalSummary],
        ablation_configs: List[AblationConfig],
    ) -> str:
        """Generate ASCII table comparing ablation study results."""
        baseline_name = "BM25 Only (Baseline)"
        baseline_summary = results.get(baseline_name)
        baseline_recall = baseline_summary.recall_at_k if baseline_summary else None

        lines = []
        lines.append("┌─────────────────────────┬──────────┬──────────┬──────────┬─────────────┬─────────┐")
        lines.append("│ Configuration           │ Recall@K │ MRR      │ NDCG@K   │ Latency(ms) │ vs Base │")
        lines.append("├─────────────────────────┼──────────┼──────────┼──────────┼─────────────┼─────────┤")

        for config in ablation_configs:
            if config.name not in results:
                continue

            summary = results[config.name]
            recall = f"{summary.recall_at_k:.4f}"
            mrr = f"{summary.mrr:.4f}"
            ndcg = f"{summary.ndcg_at_k:.4f}"
            latency = f"{summary.avg_latency_ms:.1f}"

            vs_base = "   --   "
            if baseline_recall and baseline_recall > 0 and config.name != baseline_name:
                imp_pct = (summary.recall_at_k - baseline_recall) / baseline_recall * 100
                vs_base = f"{imp_pct:+.1f}%".rjust(8)

            name_display = config.name[:23].ljust(23)

            lines.append(
                f"│ {name_display} │ {recall:8} │ {mrr:8} │ {ndcg:8} │ {latency:11} │ {vs_base:7} │"
            )

        lines.append("└─────────────────────────┴──────────┴──────────┴──────────┴─────────────┴─────────┘")
        return "\n".join(lines)

    @staticmethod
    def generate_summary_table(summary: EvalSummary, include_baseline: bool = True) -> str:
        """Generate a summary table for a single evaluation run."""
        k = summary.config.top_k
        lines = []

        lines.append("\n" + "=" * 60)
        lines.append("RAG Evaluation Results")
        lines.append("=" * 60)
        lines.append(f"\nDataset: {summary.dataset_name}")
        lines.append(f"Queries: {summary.num_queries}")
        if summary.random_seed is not None:
            lines.append(f"Random Seed: {summary.random_seed}")

        # Main metrics table
        lines.append("\n" + "-" * 60)
        lines.append(f"{'':15} {'RAG Pipeline':>15} {'BM25 Baseline':>15}")
        lines.append("-" * 60)
        lines.append(f"{'Recall@' + str(k):15} {summary.recall_at_k:>15.4f} {summary.baseline_recall_at_k or 0:>15.4f}")
        lines.append(f"{'MRR':15} {summary.mrr:>15.4f} {summary.baseline_mrr or 0:>15.4f}")
        lines.append(f"{'Hit Rate@' + str(k):15} {summary.hit_rate_at_k:>15.4f} {summary.baseline_ndcg_at_k or 0:>15.4f}")
        lines.append(f"{'NDCG@' + str(k):15} {summary.ndcg_at_k:>15.4f} {summary.baseline_ndcg_at_k or 0:>15.4f}")
        lines.append(f"{'Avg Latency (ms)':15} {summary.avg_latency_ms:>15.2f} {'--':>15}")
        lines.append("-" * 60)

        # Calculate improvement
        if include_baseline and summary.baseline_recall_at_k and summary.baseline_recall_at_k > 0:
            recall_imp = (summary.recall_at_k - summary.baseline_recall_at_k) / summary.baseline_recall_at_k * 100
            lines.append(f"\nImprovement: Recall {recall_imp:+.1f}%")

        # Question type breakdown
        if summary.question_type_breakdown:
            lines.append("\n--- By Question Type ---")
            for qtype, data in summary.question_type_breakdown.items():
                lines.append(f"  {qtype}: Recall={data['recall']:.4f}, MRR={data['mrr']:.4f} ({data['total']} queries)")

        return "\n".join(lines)

    def generate_full_report(
        self,
        summary: EvalSummary,
        queries: List[EvalQuery],
        ablation_results: Optional[Dict[str, EvalSummary]] = None,
        ablation_configs: Optional[List[AblationConfig]] = None,
        show_failure_analysis: bool = False,
    ) -> str:
        """Generate a complete report with all sections."""
        parts = []

        parts.append(self.generate_summary_table(summary))

        if ablation_results and ablation_configs:
            parts.append("\n\n" + "=" * 60)
            parts.append("Ablation Study Results")
            parts.append("=" * 60)
            parts.append("\n" + self.generate_ablation_table(ablation_results, ablation_configs))

        if show_failure_analysis and summary.details:
            failures = [r for r in summary.details if not r.hit]
            if failures:
                query_map = {q.id: q for q in queries}
                analyses = [
                    self.failure_analyzer.analyze_failure(f, query_map.get(f.query_id, EvalQuery(
                        id=f.query_id, query=f.query, golden_context=""
                    )))
                    for f in failures
                ]
                suggestions = self.failure_analyzer.generate_improvement_suggestions(analyses)
                parts.append("\n--- IMPROVEMENT SUGGESTIONS ---")
                for i, suggestion in enumerate(suggestions, 1):
                    parts.append(f"\n{i}. {suggestion}")

        return "\n".join(parts)
