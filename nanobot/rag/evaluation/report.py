"""RAG Evaluation - Report Generator.

This module provides functionality to generate beautiful ASCII reports
for RAG evaluation results, including ablation study comparisons and
failure case analysis.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from nanobot.rag.evaluation.ablation import AblationConfig
from nanobot.rag.evaluation.base import EvalQuery, EvalResult, EvalSummary


class FailureCategory(Enum):
    """Categories for classifying retrieval failures."""
    CHUNKING_ISSUE = "chunking_issue"           # Golden content split across chunks
    RETRIEVAL_BM25_FAIL = "bm25_fail"           # BM25 didn't retrieve the chunk
    RETRIEVAL_VECTOR_FAIL = "vector_fail"        # Vector search didn't retrieve the chunk
    RERANKING_FAIL = "reranking_fail"            # Chunk was reranked out of top-K
    THRESHOLD_FILTERED = "threshold_filtered"    # Chunk was filtered by threshold
    SEMANTIC_MISMATCH = "semantic_mismatch"      # Semantic similarity too low
    UNKNOWN = "unknown"                           # Could not determine


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
        """Classify the reason for a retrieval failure.

        Args:
            result: The evaluation result for the query
            query: The original evaluation query

        Returns:
            FailureAnalysis with category and description
        """
        if result.hit:
            return FailureAnalysis(
                category=FailureCategory.UNKNOWN,
                description="Query succeeded, no failure to analyze"
            )

        # Analyze based on failure_reason from ResultJudge
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

        # Default: unknown
        return FailureAnalysis(
            category=FailureCategory.UNKNOWN,
            description=f"Unclassified failure: {failure_reason}",
            suggested_fix="Manual investigation recommended"
        )

    def generate_improvement_suggestions(
        self,
        failures: List[FailureAnalysis],
    ) -> List[str]:
        """Generate improvement suggestions based on failure distribution.

        Args:
            failures: List of failure analyses

        Returns:
            List of suggestion strings
        """
        if not failures:
            return ["No failures to analyze!"]

        # Count failure categories
        category_counts: Dict[FailureCategory, int] = {}
        for failure in failures:
            category_counts[failure.category] = category_counts.get(failure.category, 0) + 1

        suggestions = []
        total = len(failures)

        # Suggest based on most common failures
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total) * 100
            if percentage < 5:
                continue  # Skip rare failures

            if category == FailureCategory.CHUNKING_ISSUE:
                suggestions.append(
                    f"🔧 Chunking issues ({percentage:.1f}%): Consider adjusting chunk size "
                    f"(current issues affect {count} queries)"
                )
            elif category == FailureCategory.RETRIEVAL_BM25_FAIL:
                suggestions.append(
                    f"🔍 BM25 retrieval failures ({percentage:.1f}%): Try lowering bm25_threshold "
                    f"or increasing recall_bm25_top_k (affects {count} queries)"
                )
            elif category == FailureCategory.RETRIEVAL_VECTOR_FAIL:
                suggestions.append(
                    f"🎯 Vector retrieval failures ({percentage:.1f}%): Try lowering vector_threshold "
                    f"or increasing recall_vector_top_k (affects {count} queries)"
                )
            elif category == FailureCategory.RERANKING_FAIL:
                suggestions.append(
                    f"📊 Reranking failures ({percentage:.1f}%): Check rerank_threshold or consider "
                    f"a different reranker model (affects {count} queries)"
                )
            elif category == FailureCategory.THRESHOLD_FILTERED:
                suggestions.append(
                    f"🚫 Threshold filtered ({percentage:.1f}%): Consider relaxing thresholds "
                    f"(affects {count} queries)"
                )
            elif category == FailureCategory.SEMANTIC_MISMATCH:
                suggestions.append(
                    f"🧠 Semantic mismatch ({percentage:.1f}%): Consider query expansion or "
                    f"different embedding model (affects {count} queries)"
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
        """Generate ASCII table comparing ablation study results.

        Args:
            results: Dictionary mapping config names to EvalSummary
            ablation_configs: List of ablation configurations used

        Returns:
            Formatted ASCII table string
        """
        # Get baseline for comparison
        baseline_name = "BM25 Only (Baseline)"
        baseline_summary = results.get(baseline_name)
        baseline_recall = baseline_summary.recall_at_5 if baseline_summary else None

        # Build table header
        lines = []
        lines.append("┌─────────────────────────┬──────────┬──────────┬─────────────┬─────────┐")
        lines.append("│ Configuration           │ Recall@5 │ MRR      │ Latency(ms) │ vs Base │")
        lines.append("├─────────────────────────┼──────────┼──────────┼─────────────┼─────────┤")

        # Add rows for each config in order
        for config in ablation_configs:
            if config.name not in results:
                continue

            summary = results[config.name]
            recall = f"{summary.recall_at_5:.4f}"
            mrr = f"{summary.mrr:.4f}"
            latency = f"{summary.avg_latency_ms:.1f}"

            # Calculate improvement vs baseline
            vs_base = "   --   "
            if baseline_recall and baseline_recall > 0 and config.name != baseline_name:
                imp_pct = (summary.recall_at_5 - baseline_recall) / baseline_recall * 100
                vs_base = f"{imp_pct:+.1f}%".rjust(8)

            # Truncate config name if too long
            name_display = config.name[:23].ljust(23)

            lines.append(
                f"│ {name_display} │ {recall:8} │ {mrr:8} │ {latency:11} │ {vs_base:7} │"
            )

        lines.append("└─────────────────────────┴──────────┴──────────┴─────────────┴─────────┘")
        return "\n".join(lines)

    @staticmethod
    def generate_failure_analysis(
        summary: EvalSummary,
        queries: List[EvalQuery],
        max_examples: int = 10,
    ) -> str:
        """Generate detailed failure analysis report.

        Args:
            summary: Evaluation summary with detailed results
            queries: Original evaluation queries
            max_examples: Maximum number of failure examples to show

        Returns:
            Formatted failure analysis string
        """
        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("FAILURE ANALYSIS")
        lines.append("=" * 60)

        if not summary.details:
            lines.append("No detailed results available")
            return "\n".join(lines)

        # Build query map
        query_map = {q.id: q for q in queries}

        # Separate failures
        failures = [r for r in summary.details if not r.hit]
        successes = [r for r in summary.details if r.hit]

        lines.append(f"\nTotal Queries: {len(summary.details)}")
        lines.append(f"Successes: {len(successes)} ({len(successes)/len(summary.details)*100:.1f}%)")
        lines.append(f"Failures: {len(failures)} ({len(failures)/len(summary.details)*100:.1f}%)")

        # Difficulty breakdown for failures
        if failures:
            lines.append("\n--- Failure by Difficulty ---")
            diff_counts: Dict[str, int] = {}
            for f in failures:
                diff = f.difficulty or "unknown"
                diff_counts[diff] = diff_counts.get(diff, 0) + 1

            for diff, count in sorted(diff_counts.items()):
                lines.append(f"  {diff}: {count}")

        # Failure reason breakdown
        if failures:
            lines.append("\n--- Failure Reasons ---")
            reason_counts: Dict[str, int] = {}
            for f in failures:
                reason = f.failure_reason or "unknown"
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

            for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  {reason}: {count}")

        # Show example failures
        if failures:
            lines.append(f"\n--- Example Failures (first {min(max_examples, len(failures))}) ---")
            for i, result in enumerate(failures[:max_examples]):
                query = query_map.get(result.query_id)
                lines.append(f"\n[{i+1}] Query: {result.query}")
                if query:
                    lines.append(f"      Source: {query.source_doc or 'unknown'}")
                    lines.append(f"      Expected chunk ID: {query.source_chunk_id}")
                if result.failure_reason:
                    lines.append(f"      Reason: {result.failure_reason}")
                if result.found_chunk_ids:
                    lines.append(f"      Found chunks: {result.found_chunk_ids}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    @staticmethod
    def generate_summary_table(summary: EvalSummary, include_baseline: bool = True) -> str:
        """Generate a summary table for a single evaluation run.

        Args:
            summary: Evaluation summary to display
            include_baseline: Whether to include baseline comparison

        Returns:
            Formatted ASCII table
        """
        lines = []

        # Core metrics
        lines.append("\n" + "=" * 50)
        lines.append("EVALUATION SUMMARY")
        lines.append("=" * 50)
        lines.append(f"\nDataset: {summary.dataset_name}")
        lines.append(f"Queries: {summary.num_queries}")
        if summary.random_seed is not None:
            lines.append(f"Random Seed: {summary.random_seed}")

        # Table
        lines.append("\n┌─────────────┬───────────┐")
        lines.append("│ Metric      │ Value     │")
        lines.append("├─────────────┼───────────┤")
        lines.append(f"│ Recall@5    │ {summary.recall_at_5:.4f}    │")
        lines.append(f"│ MRR         │ {summary.mrr:.4f}    │")
        lines.append(f"│ Hit Rate@5  │ {summary.hit_rate_at_5:.4f}    │")
        lines.append(f"│ Avg Latency │ {summary.avg_latency_ms:7.2f}ms │")
        lines.append("└─────────────┴───────────┘")

        # Baseline comparison
        if include_baseline and summary.baseline_recall_at_5 is not None:
            lines.append("\n--- Baseline Comparison ---")
            lines.append(f"Baseline Recall@5: {summary.baseline_recall_at_5:.4f}")
            lines.append(f"Baseline MRR: {summary.baseline_mrr:.4f}")
            if summary.baseline_recall_at_5 > 0:
                imp = (summary.recall_at_5 - summary.baseline_recall_at_5) / summary.baseline_recall_at_5 * 100
                lines.append(f"Improvement: {imp:+.1f}%")

        # Difficulty breakdown
        if summary.difficulty_breakdown:
            lines.append("\n--- Difficulty Breakdown ---")
            for diff, data in summary.difficulty_breakdown.items():
                lines.append(f"  {diff}: {data['hits']}/{data['total']} ({data['recall']:.1%})")

        # Failure breakdown
        if summary.failure_breakdown:
            lines.append("\n--- Failure Breakdown ---")
            for reason, count in summary.failure_breakdown.items():
                lines.append(f"  {reason}: {count}")

        return "\n".join(lines)

    def generate_full_report(
        self,
        summary: EvalSummary,
        queries: List[EvalQuery],
        ablation_results: Optional[Dict[str, EvalSummary]] = None,
        ablation_configs: Optional[List[AblationConfig]] = None,
        show_failure_analysis: bool = False,
    ) -> str:
        """Generate a complete report with all sections.

        Args:
            summary: Main evaluation summary
            queries: Original evaluation queries
            ablation_results: Optional ablation study results
            ablation_configs: Optional ablation configurations
            show_failure_analysis: Whether to show detailed failure analysis

        Returns:
            Complete formatted report
        """
        parts = []

        # Main summary
        parts.append(self.generate_summary_table(summary))

        # Ablation table if available
        if ablation_results and ablation_configs:
            parts.append("\n\n" + "=" * 60)
            parts.append("ABLATION STUDY RESULTS")
            parts.append("=" * 60)
            parts.append("\n" + self.generate_ablation_table(ablation_results, ablation_configs))

            # Add descriptions
            parts.append("\n\n--- Configurations ---")
            for config in ablation_configs:
                parts.append(f"\n{config.name}:")
                parts.append(f"  {config.description}")

        # Failure analysis if requested
        if show_failure_analysis:
            parts.append(self.generate_failure_analysis(summary, queries))

            # Improvement suggestions
            if summary.details:
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
