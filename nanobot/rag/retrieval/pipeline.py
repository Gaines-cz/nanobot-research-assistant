"""Advanced search pipeline combining multiple retrieval stages.

This module provides the multi-step advanced search pipeline:
1. Core chunk recall (BM25 + vector, dual thresholds, TopK)
2. Context expansion (prev1 + core + next1)
3. Document-level prioritization (Top3 docs)
4. Cross-Encoder rerank + semantic dedup
"""

import hashlib
import time
from typing import Any, List, Optional, Tuple

from loguru import logger

from nanobot.config.schema import RAGConfig
from nanobot.rag.embeddings import EmbeddingProvider
from nanobot.rag.models import (
    ChunkInfo,
    DocumentInfo,
    SearchResult,
    SearchResultWithContext,
)
from nanobot.rag.retrieval.base import AdvancedRetriever
from nanobot.rag.retrieval.bm25 import BM25Retriever
from nanobot.rag.retrieval.context import ContextExpander
from nanobot.rag.retrieval.hybrid import HybridRetriever
from nanobot.rag.retrieval.rerank import CrossEncoderReranker, RerankService
from nanobot.rag.retrieval.vector import VectorRetriever
from nanobot.rag.storage.connection import DatabaseConnection


class AdvancedSearchPipeline(AdvancedRetriever):
    """
    Advanced multi-step search pipeline.

    Orchestrates the complete search workflow:
    1. Core chunk recall with BM25 + vector
    2. Context expansion
    3. Document-level prioritization
    4. Cross-Encoder reranking + semantic deduplication
    """

    def __init__(
        self,
        db_connection: DatabaseConnection,
        embedding_provider: Optional[EmbeddingProvider] = None,
        config: Optional[RAGConfig] = None,
    ):
        """
        Initialize advanced search pipeline.

        Args:
            db_connection: Database connection manager
            embedding_provider: Optional embedding provider
            config: RAG configuration
        """
        super().__init__(db_connection, embedding_provider, config)

        # Create component retrievers
        self._vector_retriever = VectorRetriever(db_connection, embedding_provider, config)
        self._bm25_retriever = BM25Retriever(db_connection, embedding_provider, config)
        self._hybrid_retriever = HybridRetriever(db_connection, embedding_provider, config)
        self._context_expander = ContextExpander(db_connection, config)

        # Initialize rerank service (always create, checks config at runtime)
        self._rerank_service: Optional[RerankService] = None
        if embedding_provider is not None:
            self._rerank_service = RerankService(
                config=self.config
            )

    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Basic hybrid search (delegates to HybridRetriever logic).

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of SearchResult sorted by relevance
        """
        # Delegate to HybridRetriever to avoid code duplication
        return await self._hybrid_retriever.search(query, top_k)

    async def search_advanced(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[SearchResultWithContext]:
        """
        Advanced multi-step search pipeline:
        1. Core chunk recall (BM25 + vector, dual thresholds, TopK)
        2. Context expansion (prev1 + core + next1)
        3. Document-level prioritization (Top3 docs)
        4. Cross-Encoder rerank + semantic dedup

        Args:
            query: Search query
            top_k: Number of results to return (default: None, uses pipeline defaults)
        """
        # Validate top_k parameter
        if top_k is not None:
            if top_k < 0:
                logger.warning("[RAG] top_k cannot be negative, using default (5)")
                top_k = 5
            elif top_k == 0:
                logger.debug("[RAG] top_k is 0, returning empty results")
                if self.config.enable_search_cache:
                    cache_key = self._get_cache_key(query, 0)
                    self._cache_manager.advanced.set(cache_key, [])
                return []

        # Use default top_k for internal pipeline steps if not specified
        internal_top_k = top_k if top_k is not None else 5

        # Check cache first - use internal_top_k for cache key when top_k is None
        cache_key_top_k = top_k if top_k is not None else internal_top_k
        cache_key = self._get_cache_key(query, cache_key_top_k)
        if self.config.enable_search_cache:
            cached_results = self._cache_manager.advanced.get(cache_key)
            if cached_results is not None:
                logger.warning("[RAG CACHE] ⚠️ CACHE HIT for query: {}", query[:50])
                return cached_results
            else:
                logger.info("[RAG CACHE] Cache miss for query: {} (cache enabled)", query[:50])
        else:
            logger.info("[RAG CACHE] Cache disabled for query: {} - will execute fresh search", query[:50])

        # Log search entry
        logger.info("[RAG] search_advanced called with query: {}", query)
        logger.info("[RAG] Config - bm25_threshold: {}, vector_threshold: {}, top_k: {}, vector_weight: {}, bm25_weight: {}",
                    self.config.bm25_threshold, self.config.vector_threshold,
                    self.config.top_k, self.config.vector_weight, self.config.bm25_weight)

        # Track overall search performance
        overall_start = time.perf_counter()

        # Expand query
        expanded_query = self._query_expander.expand(query)
        if expanded_query != query:
            logger.info("[RAG] Expanded query: {}", expanded_query)

        # Step 1-3: Core recall -> Context expansion -> Document-level -> Merge
        step1_start = time.perf_counter()
        core_results = await self._step1_core_chunk_recall(expanded_query, internal_top_k)
        step1_elapsed = (time.perf_counter() - step1_start) * 1000
        logger.info("[RAG PERF] Step1 core chunk recall: {} results, elapsed={:.1f}ms",
                    len(core_results), step1_elapsed)

        if not core_results:
            logger.info("[RAG] No core results found")
            overall_elapsed = (time.perf_counter() - overall_start) * 1000
            logger.info("[RAG PERF] search_advanced total: elapsed={:.1f}ms (no results)", overall_elapsed)
            if self.config.enable_search_cache:
                self._cache_manager.advanced.set(cache_key, [])
            return []

        logger.info("[RAG] Got {} core results", len(core_results))

        step2_start = time.perf_counter()
        expanded_chunks = self._step2_context_expansion(core_results)
        step2_elapsed = (time.perf_counter() - step2_start) * 1000
        logger.info("[RAG PERF] Step2 context expansion: {} chunks, elapsed={:.1f}ms",
                    len(expanded_chunks), step2_elapsed)

        step3_start = time.perf_counter()
        top_docs = self._step3_document_level(core_results)
        step3_elapsed = (time.perf_counter() - step3_start) * 1000
        logger.info("[RAG PERF] Step3 document level: {} docs, elapsed={:.1f}ms",
                    len(top_docs), step3_elapsed)

        merge_start = time.perf_counter()
        merged_results = self._merge_context_and_document_results(expanded_chunks, top_docs)
        merge_elapsed = (time.perf_counter() - merge_start) * 1000
        logger.info("[RAG PERF] Merge results: {} merged, elapsed={:.1f}ms",
                    len(merged_results), merge_elapsed)

        # Step 4: Apply rerank and dedup
        if self.config.enable_rerank and self._rerank_service:
            rerank_start = time.perf_counter()
            final_results = await self._apply_rerank(expanded_query, merged_results)
            rerank_elapsed = (time.perf_counter() - rerank_start) * 1000
            logger.info("[RAG PERF] Step4 rerank+dedup: elapsed={:.1f}ms", rerank_elapsed)
        else:
            final_results = merged_results

        # Truncate to top_k if specified (before caching, so cache stores exactly what we return)
        if top_k is not None:
            final_results = final_results[:top_k]

        logger.info("[RAG] Final results count: {}", len(final_results))

        # Log overall performance
        overall_elapsed = (time.perf_counter() - overall_start) * 1000
        logger.info("[RAG PERF] search_advanced total: elapsed={:.1f}ms, results={}",
                    overall_elapsed, len(final_results))

        # Store in cache
        if self.config.enable_search_cache:
            self._cache_manager.advanced.set(cache_key, final_results)

        return final_results

    async def _step1_core_chunk_recall(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Step 1: Core chunk recall using BM25 + vector search.

        Respects enable_bm25 and enable_vector config options for ablation studies.
        """
        bm25_threshold = self.config.bm25_threshold
        vector_threshold = self.config.vector_threshold

        logger.info("[RAG] _step1_core_chunk_recall starting for query: {}", query)
        logger.info("[RAG] Config - enable_bm25: {}, enable_vector: {}",
                    self.config.enable_bm25, self.config.enable_vector)
        logger.info("[RAG] Thresholds - vector: {}, bm25: {}", vector_threshold, bm25_threshold)

        # Get vector results (if enabled)
        vector_results = []
        vector_elapsed = 0.0
        if self._db.vector_enabled and self.config.enable_vector:
            vector_start = time.perf_counter()
            vector_results = await self._vector_retriever.search(query, self.config.recall_vector_top_k)
            vector_elapsed = (time.perf_counter() - vector_start) * 1000
            if vector_results:
                v_scores = [r.score for r in vector_results]
                logger.info("[RAG] Vector search: {} results, scores min={:.4f}, max={:.4f}, avg={:.4f}, elapsed={:.1f}ms",
                            len(vector_results), min(v_scores), max(v_scores), sum(v_scores)/len(v_scores), vector_elapsed)
            else:
                logger.info("[RAG] Vector search: 0 results, elapsed={:.1f}ms", vector_elapsed)
        elif not self.config.enable_vector:
            logger.info("[RAG] Vector search disabled by config")

        # Get BM25 results (if enabled)
        fulltext_results = []
        bm25_elapsed = 0.0
        if self.config.enable_bm25:
            bm25_start = time.perf_counter()
            fulltext_results = self._bm25_retriever._fulltext_search(query, self.config.recall_bm25_top_k)
            bm25_elapsed = (time.perf_counter() - bm25_start) * 1000
            if fulltext_results:
                ft_scores = [r.score for r in fulltext_results]
                logger.info("[RAG] Full-text search: {} results, scores min={:.4f}, max={:.4f}, avg={:.4f}, elapsed={:.1f}ms",
                            len(fulltext_results), min(ft_scores), max(ft_scores), sum(ft_scores)/len(ft_scores), bm25_elapsed)
            else:
                logger.info("[RAG] Full-text search: 0 results, elapsed={:.1f}ms", bm25_elapsed)
        else:
            logger.info("[RAG] BM25 search disabled by config")

        # If both are disabled, return empty
        if not self.config.enable_bm25 and not self.config.enable_vector:
            logger.warning("[RAG] Both BM25 and Vector search are disabled!")
            return []

        # If only one is enabled, use it directly
        if not self.config.enable_bm25:
            logger.info("[RAG] Using only vector search results")
            filtered_vector = [r for r in vector_results if r.score >= vector_threshold] if vector_results else []
            if len(filtered_vector) < 3 and vector_results:
                filtered_vector = vector_results[:max(top_k, 10)]
            return filtered_vector[:self.config.recall_stage1_top_k]

        if not self.config.enable_vector or not self._db.vector_enabled:
            logger.info("[RAG] Using only BM25 search results")
            filtered_ft = [r for r in fulltext_results if r.score >= bm25_threshold] if fulltext_results else []
            if len(filtered_ft) < 3 and fulltext_results:
                filtered_ft = fulltext_results[:max(top_k, 10)]
            return filtered_ft[:self.config.recall_stage1_top_k]

        # Continue with hybrid search (both enabled)
        try:
            # Use soft filtering: apply thresholds but keep a minimum number of results
            # Step 1: Apply primary vector threshold (0.72)
            filtered_vector = [r for r in vector_results if r.score >= vector_threshold]
            filtered_ft = [r for r in fulltext_results if r.score >= bm25_threshold]

            # Step 2: If too few vector results, relax to 0.6 threshold
            min_recall_candidates = 10
            if len(filtered_vector) < 3 and vector_results:
                relax_threshold = 0.6
                logger.info("[RAG] Too few vector results ({}), relaxing threshold to {}", len(filtered_vector), relax_threshold)
                filtered_vector = [r for r in vector_results if r.score >= relax_threshold]

                # Step 3: If still too few, take top candidates directly
                if len(filtered_vector) < 3:
                    logger.info("[RAG] Still too few vector results ({}), taking top candidates", len(filtered_vector))
                    filtered_vector = vector_results[:max(top_k, min_recall_candidates)]

            if len(filtered_ft) < 3 and fulltext_results:
                logger.info("[RAG] Too few fulltext results ({}), relaxing threshold", len(filtered_ft))
                filtered_ft = fulltext_results[:max(top_k, min_recall_candidates)]  # Keep more candidates for recall

            logger.info("[RAG] After relaxed filtering: vector={}, fulltext={}",
                        len(filtered_vector), len(filtered_ft))

            logger.info("[RAG] Step1 percentile-weighted fusion starting - filtered_ft={}, filtered_vector={}",
                        len(filtered_ft), len(filtered_vector))

            # Percentile-weighted fusion
            fusion_start = time.perf_counter()
            VECTOR_WEIGHT = self.config.vector_weight
            BM25_WEIGHT = self.config.bm25_weight

            n_bm25 = len(filtered_ft)
            n_vector = len(filtered_vector)
            weighted_scores: dict[str, tuple[SearchResult, float]] = {}

            # BM25 contributes (percentile = (n - rank + 1) / n)
            for rank, r in enumerate(filtered_ft, start=1):
                key = f"{r.path}:{r.chunk_index}"
                bm25_percentile = (n_bm25 - rank + 1) / n_bm25 if n_bm25 > 0 else 0
                if key not in weighted_scores:
                    weighted_scores[key] = (r, BM25_WEIGHT * bm25_percentile)
                else:
                    weighted_scores[key] = (weighted_scores[key][0],
                                              weighted_scores[key][1] + BM25_WEIGHT * bm25_percentile)

            # Vector contributes (percentile = (n - rank + 1) / n)
            for rank, r in enumerate(filtered_vector, start=1):
                key = f"{r.path}:{r.chunk_index}"
                vector_percentile = (n_vector - rank + 1) / n_vector if n_vector > 0 else 0
                if key not in weighted_scores:
                    weighted_scores[key] = (r, VECTOR_WEIGHT * vector_percentile)
                else:
                    weighted_scores[key] = (weighted_scores[key][0],
                                              weighted_scores[key][1] + VECTOR_WEIGHT * vector_percentile)

            # Sort by weighted score descending and take top-k
            sorted_results = sorted(weighted_scores.values(), key=lambda x: x[1], reverse=True)
            final_results = [r for r, _ in sorted_results[:self.config.recall_stage1_top_k]]
            fusion_elapsed = (time.perf_counter() - fusion_start) * 1000
            final_sources = [r.source for r in final_results]
            final_scores = [f"{r.score:.4f}" for r in final_results]
            logger.info("[RAG PERF] Step1 fusion: {} results, sources={}, scores={}, elapsed={:.1f}ms",
                        len(final_results), final_sources, final_scores, fusion_elapsed)

            return final_results
        except Exception as e:
            logger.warning("Hybrid search failed, falling back: {}", e)
            # Fallback to BM25 only if enabled
            if self.config.enable_bm25:
                fallback_start = time.perf_counter()
                logger.info("[RAG] Hybrid failed, using BM25 fallback")
                fulltext_results = self._bm25_retriever._fulltext_search(query, top_k)
                fallback_elapsed = (time.perf_counter() - fallback_start) * 1000
                final_results = fulltext_results[:top_k]
                logger.info("[RAG PERF] Fallback: {} results, elapsed={:.1f}ms", len(final_results), fallback_elapsed)
                return final_results
            return []

    def _step2_context_expansion(self, core_results: List[SearchResult]) -> List[ChunkInfo]:
        """Step 2: Expand context around core chunks."""
        return self._context_expander.expand_context(
            core_results,
            enable_context_expansion=self.config.enable_context_expansion
        )

    def _step3_document_level(self, core_results: List[SearchResult]) -> List[Tuple[DocumentInfo, float]]:
        """Step 3: Document-level retrieval."""
        step3_start = time.perf_counter()

        if not self.config.enable_document_level:
            return []

        db = self._db.db
        doc_scores: dict[int, List[float]] = {}
        doc_info_map: dict[int, DocumentInfo] = {}

        for result in core_results:
            cursor = db.execute("""
                SELECT id, path, filename, file_type, file_size, title, doc_type, abstract
                FROM documents WHERE path = ?
            """, (result.path,))
            row = cursor.fetchone()
            if row:
                doc_id = row[0]
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = []
                    doc_info_map[doc_id] = DocumentInfo(
                        id=row[0],
                        path=row[1],
                        filename=row[2],
                        file_type=row[3],
                        file_size=row[4],
                        title=row[5],
                        doc_type=row[6],
                        abstract=row[7],
                    )
                doc_scores[doc_id].append(result.score)

        doc_avg_scores = []
        for doc_id, scores in doc_scores.items():
            avg_score = sum(scores) / len(scores)
            doc_avg_scores.append((doc_info_map[doc_id], avg_score))

        doc_avg_scores.sort(key=lambda x: x[1], reverse=True)
        result = doc_avg_scores[:self.config.top_documents]

        step3_elapsed = (time.perf_counter() - step3_start) * 1000
        logger.info("[RAG PERF] Step3 document level: {} docs, elapsed={:.1f}ms",
                    len(result), step3_elapsed)

        return result

    def _merge_context_and_document_results(
        self,
        expanded_chunks: List[ChunkInfo],
        top_docs: List[Tuple[DocumentInfo, float]]
    ) -> List[SearchResultWithContext]:
        """Merge context-expanded chunks with document-level results."""
        merge_start = time.perf_counter()

        db = self._db.db
        results = []

        doc_info_cache: dict[int, DocumentInfo] = {}
        for doc, _ in top_docs:
            doc_info_cache[doc.id] = doc

        for i, chunk in enumerate(expanded_chunks):
            if chunk.doc_id not in doc_info_cache:
                cursor = db.execute("""
                    SELECT id, path, filename, file_type, file_size, title, doc_type, abstract
                    FROM documents WHERE id = ?
                """, (chunk.doc_id,))
                row = cursor.fetchone()
                if row:
                    doc_info_cache[chunk.doc_id] = DocumentInfo(
                        id=row[0],
                        path=row[1],
                        filename=row[2],
                        file_type=row[3],
                        file_size=row[4],
                        title=row[5],
                        doc_type=row[6],
                        abstract=row[7],
                    )

            doc_info = doc_info_cache.get(chunk.doc_id)
            if not doc_info:
                continue

            combined_parts = []
            if chunk.prev_content:
                combined_parts.append(chunk.prev_content)
            combined_parts.append(chunk.content)
            if chunk.next_content:
                combined_parts.append(chunk.next_content)
            combined_content = "\n\n".join(combined_parts)

            doc_bonus = 0.0
            for doc, score in top_docs:
                if doc.id == chunk.doc_id:
                    doc_bonus = score * 0.1
                    break

            final_score = chunk.score + doc_bonus

            results.append(SearchResultWithContext(
                document=doc_info,
                chunk=chunk,
                combined_content=combined_content,
                final_score=final_score,
                rank=i + 1,
            ))

        results.sort(key=lambda x: x.final_score, reverse=True)
        for i, r in enumerate(results):
            r.rank = i + 1

        merge_elapsed = (time.perf_counter() - merge_start) * 1000
        logger.info("[RAG PERF] Merge results: {} merged, elapsed={:.1f}ms",
                    len(results), merge_elapsed)

        return results

    async def _apply_rerank(
        self,
        query: str,
        results: List[SearchResultWithContext],
    ) -> List[SearchResultWithContext]:
        """Apply Cross-Encoder reranking."""
        rerank_start = time.perf_counter()

        if not self._rerank_service or not results:
            return results

        try:
            candidates: List[Tuple[str, Any, List[float]]] = []
            result_map: dict[int, SearchResultWithContext] = {}

            for i, result in enumerate(results):
                candidates.append((
                    result.combined_content,
                    result,
                    []
                ))
                result_map[i] = result

            # Get embeddings for semantic deduplication (only if enabled)
            if self._db.vector_enabled and self.config.enable_rerank_dedup_embedding:
                try:
                    embed_batch_start = time.perf_counter()
                    texts = [c[0] for c in candidates]
                    embeddings = await self._embedding_provider.embed_batch(texts)
                    embed_batch_elapsed = (time.perf_counter() - embed_batch_start) * 1000
                    logger.info("[RAG PERF] Rerank embed_batch: {} texts, elapsed={:.1f}ms",
                               len(texts), embed_batch_elapsed)
                    candidates = [
                        (c[0], c[1], emb)
                        for c, emb in zip(candidates, embeddings)
                    ]
                except Exception as e:
                    logger.warning("Could not get embeddings for rerank/dedup: {}", e)
            else:
                logger.info("[RAG] Skipping embed_batch for rerank dedup (enable_rerank_dedup_embedding=False)")

            reranked = await self._rerank_service.rerank_and_dedup(query, candidates)

            final_results: List[SearchResultWithContext] = []
            for original_idx, result, new_score in reranked:
                result.final_score = new_score
                final_results.append(result)

            for i, r in enumerate(final_results):
                r.rank = i + 1

            rerank_total_elapsed = (time.perf_counter() - rerank_start) * 1000
            logger.info("[RAG PERF] _apply_rerank total: {} -> {} results, elapsed={:.1f}ms",
                       len(results), len(final_results), rerank_total_elapsed)

            return final_results

        except Exception as e:
            logger.warning("Rerank failed, returning original results: {}", e)
            return results

    def clear_cache(self) -> None:
        """Clear all search caches (including sub-component caches)."""
        # Clear own cache
        super().clear_cache()
        # Clear sub-component caches
        if hasattr(self, '_hybrid_retriever') and self._hybrid_retriever:
            self._hybrid_retriever.clear_cache()
        if hasattr(self, '_vector_retriever') and self._vector_retriever:
            self._vector_retriever.clear_cache()
        if hasattr(self, '_bm25_retriever') and self._bm25_retriever:
            self._bm25_retriever.clear_cache()

    def _get_cache_key(self, query: str, top_k: int | None = None) -> str:
        """
        Generate safe cache key using SHA-256 hash.

        Args:
            query: Search query
            top_k: Optional top_k parameter

        Returns:
            SHA-256 hash as cache key
        """
        key_bytes = query.encode("utf-8")
        if top_k is not None:
            key_bytes += f":{top_k}".encode("utf-8")
        # Add configuration parameters that affect search results
        key_bytes += f":bm25t={self.config.bm25_threshold}".encode("utf-8")
        key_bytes += f":vectort={self.config.vector_threshold}".encode("utf-8")
        key_bytes += f":vw={self.config.vector_weight}:bw={self.config.bm25_weight}".encode("utf-8")
        key_bytes += f":enable_bm25={self.config.enable_bm25}".encode("utf-8")
        key_bytes += f":enable_vector={self.config.enable_vector}".encode("utf-8")
        key_bytes += f":rerank={self.config.enable_rerank}".encode("utf-8")
        key_bytes += f":rerankt={self.config.rerank_threshold}".encode("utf-8")
        key_bytes += f":dedupt={self.config.dedup_threshold}".encode("utf-8")
        key_bytes += f":ctxexpand={self.config.enable_context_expansion}".encode("utf-8")
        key_bytes += f":doclevel={self.config.enable_document_level}".encode("utf-8")
        key_bytes += f":queryexpand={self.config.enable_query_expand}".encode("utf-8")
        return hashlib.sha256(key_bytes).hexdigest()
