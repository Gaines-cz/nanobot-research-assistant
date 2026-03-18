"""RAG Evaluation - Main evaluator."""

import re
import time
from typing import Dict, List, Optional

from loguru import logger

from nanobot.config.schema import RAGConfig
from nanobot.rag.embeddings import EmbeddingProvider
from nanobot.rag.evaluation.ablation import AblationConfig, ABLATION_CONFIGS
from nanobot.rag.evaluation.base import (
    EvalConfig,
    EvalQuery,
    EvalResult,
    EvalSummary,
)
from nanobot.rag.evaluation.baseline import BaselineRetriever
from nanobot.rag.evaluation.judge import ResultJudge
from nanobot.rag.evaluation.metrics import MetricsCalculator
from nanobot.rag.store import DocumentStore


class RAGEvaluator:
    """
    Main RAG evaluator.

    Orchestrates the evaluation process:
    1. Load test dataset
    2. Evaluate each query
    3. Calculate metrics
    4. Generate summary
    """

    def __init__(
        self,
        doc_store: DocumentStore,
        embedding_provider: Optional[EmbeddingProvider] = None,
        config: Optional[EvalConfig] = None,
    ):
        self.doc_store = doc_store
        self.embedding_provider = embedding_provider
        self.config = config or EvalConfig()
        self.judge = ResultJudge(
            strong_threshold=self.config.strong_threshold,
            weak_threshold=self.config.weak_threshold,
            db_connection=doc_store.connection if doc_store else None,
        )
        self.metrics = MetricsCalculator()
        self.baseline = BaselineRetriever(doc_store.connection) if doc_store else None

    async def evaluate(
        self,
        queries: List[EvalQuery],
        include_baseline: bool = True,
    ) -> EvalSummary:
        """
        Execute evaluation.

        Args:
            queries: List of test queries (with precomputed embeddings)
            include_baseline: Whether to run baseline comparison

        Returns:
            EvalSummary with all results
        """
        details: List[EvalResult] = []

        logger.info("Starting evaluation of {} queries", len(queries))

        for idx, query in enumerate(queries):
            if idx % 10 == 0:
                logger.info("Evaluating query {}/{}", idx + 1, len(queries))

            result = await self._evaluate_single(
                query, include_baseline
            )
            details.append(result)

        # Calculate summary metrics
        summary = EvalSummary(
            dataset_name="custom",
            num_queries=len(queries),
            config=self.config,
            random_seed=self.config.random_seed,
            recall_at_5=self.metrics.recall_at_k(details, 5),
            mrr=self.metrics.mrr(details),
            hit_rate_at_5=self.metrics.hit_rate_at_k(details, 5),
            avg_latency_ms=self.metrics.avg_latency(details),
            details=details,
        )

        # Baseline comparison
        if include_baseline and self.baseline:
            summary.baseline_recall_at_5 = self.metrics.recall_at_k(
                details, 5, use_baseline=True
            )
            summary.baseline_mrr = self.metrics.mrr(
                details, use_baseline=True
            )

        # Extended metrics
        summary.difficulty_breakdown = self.metrics.difficulty_breakdown(details)
        summary.failure_breakdown = self.metrics.failure_breakdown(details)

        logger.info("Evaluation complete!")
        logger.info("  Recall@5: {:.4f}", summary.recall_at_5)
        logger.info("  MRR: {:.4f}", summary.mrr)
        logger.info("  Hit Rate@5: {:.4f}", summary.hit_rate_at_5)
        logger.info("  Avg Latency: {:.2f}ms", summary.avg_latency_ms)

        return summary

    async def _evaluate_single(
        self,
        query: EvalQuery,
        include_baseline: bool,
    ) -> EvalResult:
        """Evaluate a single query."""
        start_time = time.time()

        # Execute search (explicit top_k=5 for Recall@5 and Hit Rate@5 metrics)
        results = await self.doc_store.search_advanced(query.query, top_k=5)
        latency_ms = (time.time() - start_time) * 1000

        # Extract result info
        found_chunk_ids = [r.chunk.id for r in results]

        # Use cached golden_embedding
        golden_embedding = query.golden_embedding

        # Apply judge
        hit, hit_reason, failure_reason, best_similarity = self.judge.judge(
            results, query, golden_embedding
        )

        # Determine hit_rank for main results
        hit_rank = None
        if hit and query.source_chunk_id:
            # Try direct ID match first
            for i, r in enumerate(results):
                if r.chunk.id == query.source_chunk_id:
                    hit_rank = i + 1
                    break
            # If not found and we have a hit_reason, parse rank from it
            if hit_rank is None and hit_reason:
                # Parse rank from hit_reason like "id_match_rank_1" or "parent_id_match_rank_2"
                match = re.search(r'_rank_(\d+)', hit_reason)
                if match:
                    hit_rank = int(match.group(1))

        # Baseline result - use same judge logic
        baseline_hit = None
        baseline_hit_rank = None
        if include_baseline and self.baseline:
            baseline_results = await self.baseline.search_bm25(
                query.query, top_k=5
            )
            if baseline_results:
                # Apply the same judge to baseline results
                baseline_hit_result, baseline_hit_reason, _, _ = self.judge.judge(
                    baseline_results, query, golden_embedding
                )
                baseline_hit = baseline_hit_result

                # Determine baseline hit_rank
                if baseline_hit and query.source_chunk_id:
                    # Try direct ID match first
                    for i, r in enumerate(baseline_results):
                        if r.chunk.id == query.source_chunk_id:
                            baseline_hit_rank = i + 1
                            break
                    # If not found and we have a hit_reason, parse rank from it
                    if baseline_hit_rank is None and baseline_hit_reason:
                        match = re.search(r'_rank_(\d+)', baseline_hit_reason)
                        if match:
                            baseline_hit_rank = int(match.group(1))

        return EvalResult(
            query_id=query.id,
            query=query.query,
            hit=hit,
            hit_rank=hit_rank,
            hit_reason=hit_reason,
            failure_reason=failure_reason,
            similarity_scores=None,  # Can be populated later if needed
            best_similarity=best_similarity,
            found_chunk_ids=found_chunk_ids,
            latency_ms=latency_ms,
            difficulty=query.difficulty,
            baseline_hit=baseline_hit,
            baseline_hit_rank=baseline_hit_rank,
        )


class AblationStudy:
    """
    Ablation study executor for RAG pipeline.

    Systematically tests the contribution of each component by
    disabling them one by one and measuring the impact on performance.
    """

    def __init__(
        self,
        doc_store: DocumentStore,
        embedding_provider: Optional[EmbeddingProvider] = None,
        eval_config: Optional[EvalConfig] = None,
        rag_config: Optional[RAGConfig] = None,
    ):
        """
        Initialize ablation study.

        Args:
            doc_store: Document store for retrieval
            embedding_provider: Embedding provider
            eval_config: Evaluation configuration
            rag_config: RAG configuration (will be modified during study)
        """
        self.doc_store = doc_store
        self.embedding_provider = embedding_provider
        self.eval_config = eval_config or EvalConfig()
        self.rag_config = rag_config

    async def run_ablation(
        self,
        queries: List[EvalQuery],
        ablation_configs: Optional[List[AblationConfig]] = None,
        include_baseline: bool = False,
    ) -> Dict[str, EvalSummary]:
        """
        Run ablation study with specified configurations.

        Args:
            queries: List of evaluation queries
            ablation_configs: List of ablation configurations (uses predefined if None)
            include_baseline: Whether to include BM25 baseline in each run

        Returns:
            Dictionary mapping config names to EvalSummary
        """
        from loguru import logger

        if ablation_configs is None:
            ablation_configs = ABLATION_CONFIGS

        results = {}

        # Backup original RAG config
        original_config = self._backup_rag_config()

        try:
            for i, ablation_cfg in enumerate(ablation_configs):
                logger.info(
                    "Running ablation {}/{}: {}",
                    i + 1, len(ablation_configs), ablation_cfg.name
                )

                # Apply ablation configuration
                self._apply_ablation_config(ablation_cfg)

                # Clear cache to ensure fresh runs
                self.doc_store.clear_cache()

                # Run evaluation
                evaluator = RAGEvaluator(
                    self.doc_store,
                    self.embedding_provider,
                    self.eval_config
                )
                summary = await evaluator.evaluate(
                    queries,
                    include_baseline=include_baseline
                )
                results[ablation_cfg.name] = summary

                logger.info(
                    "  {} completed: Recall@5={:.4f}, MRR={:.4f}, Latency={:.2f}ms",
                    ablation_cfg.name,
                    summary.recall_at_5,
                    summary.mrr,
                    summary.avg_latency_ms
                )

        finally:
            # Restore original configuration
            self._restore_rag_config(original_config)

        return results

    def _backup_rag_config(self) -> Optional[Dict]:
        """Backup current RAG configuration."""
        if self.rag_config is None:
            return None

        return {
            "enable_bm25": self.rag_config.enable_bm25,
            "enable_vector": self.rag_config.enable_vector,
            "enable_context_expansion": self.rag_config.enable_context_expansion,
            "enable_document_level": self.rag_config.enable_document_level,
            "enable_rerank": self.rag_config.enable_rerank,
            "enable_query_expand": self.rag_config.enable_query_expand,
            "enable_search_cache": self.rag_config.enable_search_cache,
        }

    def _restore_rag_config(self, backup: Optional[Dict]) -> None:
        """Restore RAG configuration from backup."""
        if backup is None or self.rag_config is None:
            return

        self.rag_config.enable_bm25 = backup["enable_bm25"]
        self.rag_config.enable_vector = backup["enable_vector"]
        self.rag_config.enable_context_expansion = backup["enable_context_expansion"]
        self.rag_config.enable_document_level = backup["enable_document_level"]
        self.rag_config.enable_rerank = backup["enable_rerank"]
        self.rag_config.enable_query_expand = backup["enable_query_expand"]
        self.rag_config.enable_search_cache = backup["enable_search_cache"]

    def _apply_ablation_config(self, ablation_cfg: AblationConfig) -> None:
        """Apply an ablation configuration to the RAG config."""
        if self.rag_config is None:
            return

        # Apply config settings
        ablation_cfg.apply_to_rag_config(self.rag_config)

        # Disable cache for ablation studies to ensure consistent results
        self.rag_config.enable_search_cache = False
