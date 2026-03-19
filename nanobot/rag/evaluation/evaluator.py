"""RAG Evaluation - Main evaluator."""

import re
import time
from typing import Dict, List, Optional

from loguru import logger

from nanobot.rag.embeddings import EmbeddingProvider
from nanobot.rag.evaluation.ablation import AblationConfig, ABLATION_CONFIGS
from nanobot.rag.evaluation.base import EvalConfig, EvalQuery, EvalResult, EvalSummary
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
            db_connection=doc_store.connection if doc_store else None,
        )
        self.metrics = MetricsCalculator()
        self.baseline = BaselineRetriever(doc_store.connection) if doc_store else None
        self.k = self.config.top_k

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

            result = await self._evaluate_single(query, include_baseline)
            details.append(result)

        # Build queries map for breakdown calculation
        queries_map = {q.id: q for q in queries}

        # Calculate summary metrics
        summary = EvalSummary(
            dataset_name="custom",
            num_queries=len(queries),
            config=self.config,
            random_seed=self.config.random_seed,
            recall_at_k=self.metrics.recall_at_k(details, self.k),
            mrr=self.metrics.mrr(details),
            hit_rate_at_k=self.metrics.hit_rate_at_k(details, self.k),
            ndcg_at_k=self.metrics.ndcg_at_k(details, self.k),
            avg_latency_ms=self.metrics.avg_latency(details),
            question_type_breakdown=self.metrics.question_type_breakdown(details, queries_map),
            details=details,
        )

        # Baseline comparison
        if include_baseline and self.baseline:
            summary.baseline_recall_at_k = self.metrics.recall_at_k(details, self.k, use_baseline=True)
            summary.baseline_mrr = self.metrics.mrr(details, use_baseline=True)
            summary.baseline_ndcg_at_k = self.metrics.ndcg_at_k(details, self.k, use_baseline=True)

        logger.info("Evaluation complete!")
        logger.info("  Recall@{}: {:.4f}", self.k, summary.recall_at_k)
        logger.info("  MRR: {:.4f}", summary.mrr)
        logger.info("  Hit Rate@{}: {:.4f}", self.k, summary.hit_rate_at_k)
        logger.info("  NDCG@{}: {:.4f}", self.k, summary.ndcg_at_k)
        logger.info("  Avg Latency: {:.2f}ms", summary.avg_latency_ms)

        return summary

    async def _evaluate_single(
        self,
        query: EvalQuery,
        include_baseline: bool,
    ) -> EvalResult:
        """Evaluate a single query."""
        start_time = time.time()

        # Execute search
        results = await self.doc_store.search_advanced(query.query, top_k=self.k)
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
                match = re.search(r'_rank_(\d+)', hit_reason)
                if match:
                    hit_rank = int(match.group(1))

        # Baseline result
        baseline_hit = None
        baseline_hit_rank = None
        if include_baseline and self.baseline:
            baseline_results = await self.baseline.search_bm25(query.query, top_k=self.k)
            if baseline_results:
                baseline_hit_result, baseline_hit_reason, _, _ = self.judge.judge(
                    baseline_results, query, golden_embedding
                )
                baseline_hit = baseline_hit_result

                if baseline_hit and query.source_chunk_id:
                    for i, r in enumerate(baseline_results):
                        if r.chunk.id == query.source_chunk_id:
                            baseline_hit_rank = i + 1
                            break
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
            similarity_scores=None,
            best_similarity=best_similarity,
            found_chunk_ids=found_chunk_ids,
            latency_ms=latency_ms,
            baseline_hit=baseline_hit,
            baseline_hit_rank=baseline_hit_rank,
        )


class AblationStudy:
    """
    Ablation study to measure contribution of each pipeline component.

    Runs evaluation with different configurations to understand
    the impact of each RAG pipeline component.
    """

    def __init__(
        self,
        doc_store: DocumentStore,
        embedding_provider: Optional[EmbeddingProvider] = None,
        eval_config: Optional[EvalConfig] = None,
        rag_config=None,
    ):
        self.doc_store = doc_store
        self.embedding_provider = embedding_provider
        self.eval_config = eval_config or EvalConfig()
        self.rag_config = rag_config
        self.k = self.eval_config.top_k
        self.metrics = MetricsCalculator()

    async def run_ablation(
        self,
        queries: List[EvalQuery],
        ablation_configs: List[AblationConfig] = None,
        include_baseline: bool = False,
    ) -> Dict[str, EvalSummary]:
        """
        Run ablation study with multiple configurations.

        Args:
            queries: Test queries
            ablation_configs: List of configurations to test
            include_baseline: Whether to include baseline metrics

        Returns:
            Dict mapping config name to EvalSummary
        """
        if ablation_configs is None:
            ablation_configs = ABLATION_CONFIGS

        results: Dict[str, EvalSummary] = {}

        # Store original config
        original_rag_config = None
        if self.rag_config:
            original_rag_config = {
                'enable_bm25': self.rag_config.enable_bm25,
                'enable_vector': self.rag_config.enable_vector,
                'enable_query_expand': self.rag_config.enable_query_expand,
                'enable_context_expansion': self.rag_config.enable_context_expansion,
                'enable_document_level': self.rag_config.enable_document_level,
                'enable_rerank': self.rag_config.enable_rerank,
            }

        for config in ablation_configs:
            logger.info("Running ablation: {}", config.name)

            # Apply ablation config
            if self.rag_config and hasattr(config, 'apply_to_rag_config'):
                config.apply_to_rag_config(self.rag_config)

            # Evaluate with this config
            evaluator = RAGEvaluator(
                self.doc_store,
                self.embedding_provider,
                self.eval_config,
            )

            summary = await evaluator.evaluate(queries, include_baseline=False)
            summary.dataset_name = config.name
            results[config.name] = summary

        # Restore original config
        if self.rag_config and original_rag_config:
            self.rag_config.enable_bm25 = original_rag_config['enable_bm25']
            self.rag_config.enable_vector = original_rag_config['enable_vector']
            self.rag_config.enable_query_expand = original_rag_config['enable_query_expand']
            self.rag_config.enable_context_expansion = original_rag_config['enable_context_expansion']
            self.rag_config.enable_document_level = original_rag_config['enable_document_level']
            self.rag_config.enable_rerank = original_rag_config['enable_rerank']

        return results
