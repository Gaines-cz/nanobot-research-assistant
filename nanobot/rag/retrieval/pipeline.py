"""Advanced search pipeline combining multiple retrieval stages.

This module provides the multi-step advanced search pipeline:
1. Core chunk recall (BM25 + vector, dual thresholds, TopK)
2. Context expansion (prev1 + core + next1)
3. Document-level prioritization (Top3 docs)
4. Cross-Encoder rerank + semantic dedup
"""

import hashlib
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

        # Initialize rerank service
        self._rerank_service: Optional[RerankService] = None
        if self.config.enable_rerank and embedding_provider is not None:
            self._rerank_service = RerankService(
                reranker=CrossEncoderReranker(self.config.rerank_model),
                rerank_threshold=self.config.rerank_threshold,
                dedup_threshold=self.config.dedup_threshold,
                rerank_top_k=self.config.rerank_top_k,
                enable_rerank=self.config.enable_rerank,
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
                logger.debug("Advanced search cache hit for query: {}", query)
                return cached_results

        # Log search entry
        logger.info("[RAG] search_advanced called with query: {}", query)
        logger.info("[RAG] Config - bm25_threshold: {}, vector_threshold: {}, top_k: {}, rrf_k: {}",
                    self.config.bm25_threshold, self.config.vector_threshold,
                    self.config.top_k, self.config.rrf_k)

        # Expand query
        expanded_query = self._query_expander.expand(query)
        if expanded_query != query:
            logger.info("[RAG] Expanded query: {}", expanded_query)

        # Step 1-3: Core recall -> Context expansion -> Document-level -> Merge
        core_results = await self._step1_core_chunk_recall(expanded_query, internal_top_k)
        if not core_results:
            logger.info("[RAG] No core results found")
            if self.config.enable_search_cache:
                self._cache_manager.advanced.set(cache_key, [])
            return []

        logger.info("[RAG] Got {} core results", len(core_results))

        expanded_chunks = self._step2_context_expansion(core_results)
        top_docs = self._step3_document_level(core_results)
        merged_results = self._merge_context_and_document_results(expanded_chunks, top_docs)

        # Step 4: Apply rerank and dedup
        if self.config.enable_rerank and self._rerank_service:
            final_results = await self._apply_rerank(expanded_query, merged_results)
        else:
            final_results = merged_results

        # Truncate to top_k if specified (before caching, so cache stores exactly what we return)
        if top_k is not None:
            final_results = final_results[:top_k]

        logger.info("[RAG] Final results count: {}", len(final_results))

        # Store in cache
        if self.config.enable_search_cache:
            self._cache_manager.advanced.set(cache_key, final_results)

        return final_results

    async def _step1_core_chunk_recall(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Step 1: Core chunk recall using BM25 + vector search."""
        bm25_threshold = self.config.bm25_threshold
        vector_threshold = self.config.vector_threshold

        logger.info("[RAG] _step1_core_chunk_recall starting for query: {}", query)
        logger.info("[RAG] Thresholds - vector: {}, bm25: {}", vector_threshold, bm25_threshold)

        if self._db.vector_enabled:
            try:
                # Vector search: top 50 results (for supplement)
                vector_results = await self._vector_retriever.search(query, 50)
                # BM25 search: top 50 results (more for base)
                fulltext_results = self._bm25_retriever._fulltext_search(query, 50)

                # Log raw search results
                if vector_results:
                    v_scores = [r.score for r in vector_results]
                    logger.info("[RAG] Vector search: {} results, scores min={:.4f}, max={:.4f}, avg={:.4f}",
                                len(vector_results), min(v_scores), max(v_scores), sum(v_scores)/len(v_scores))
                else:
                    logger.info("[RAG] Vector search: 0 results")

                if fulltext_results:
                    ft_scores = [r.score for r in fulltext_results]
                    logger.info("[RAG] Full-text search: {} results, scores min={:.4f}, max={:.4f}, avg={:.4f}",
                                len(fulltext_results), min(ft_scores), max(ft_scores), sum(ft_scores)/len(ft_scores))
                else:
                    logger.info("[RAG] Full-text search: 0 results")

                # Use soft filtering: apply thresholds but keep a minimum number of results
                filtered_vector = [r for r in vector_results if r.score >= vector_threshold]
                filtered_ft = [r for r in fulltext_results if r.score >= bm25_threshold]

                # If too few results after filtering, relax thresholds
                if len(filtered_vector) < 3 and vector_results:
                    logger.info("[RAG] Too few vector results ({}), relaxing threshold", len(filtered_vector))
                    filtered_vector = vector_results[:top_k]  # Keep top_k regardless of threshold
                if len(filtered_ft) < 3 and fulltext_results:
                    logger.info("[RAG] Too few fulltext results ({}), relaxing threshold", len(filtered_ft))
                    filtered_ft = fulltext_results[:top_k]  # Keep top_k regardless of threshold

                logger.info("[RAG] After relaxed filtering: vector={}, fulltext={}",
                            len(filtered_vector), len(filtered_ft))

                logger.info("[RAG] Step1 StagedFusion starting - filtered_ft={}, filtered_vector={}",
                            len(filtered_ft), len(filtered_vector))

                # Stage 1: BM25 first (all of filtered_ft)
                seen = set()
                merged = []
                for r in filtered_ft:
                    key = f"{r.path}:{r.chunk_index}"
                    seen.add(key)
                    merged.append(r)

                # Stage 2: Add vector results not in BM25
                for r in filtered_vector:
                    key = f"{r.path}:{r.chunk_index}"
                    if key not in seen:
                        seen.add(key)
                        merged.append(r)

                # Take top_k
                final_results = merged[:top_k]
                final_sources = [r.source for r in final_results]
                final_scores = [f"{r.score:.4f}" for r in final_results]
                logger.info("[RAG] Step1 StagedFusion complete - final_results={}, sources={}, scores={}",
                            len(final_results), final_sources, final_scores)

                return final_results
            except Exception as e:
                logger.warning("Hybrid search failed, falling back: {}", e)

        # Fallback: full-text only
        logger.info("[RAG] Vector disabled or hybrid failed, using full-text fallback")
        fulltext_results = self._bm25_retriever._fulltext_search(query, top_k)
        # Don't apply threshold filtering for fallback
        final_results = fulltext_results[:top_k]
        logger.info("[RAG] Fallback: {} results", len(final_results))
        return final_results

    def _step2_context_expansion(self, core_results: List[SearchResult]) -> List[ChunkInfo]:
        """Step 2: Expand context around core chunks."""
        return self._context_expander.expand_context(
            core_results,
            enable_context_expansion=self.config.enable_context_expansion
        )

    def _step3_document_level(self, core_results: List[SearchResult]) -> List[Tuple[DocumentInfo, float]]:
        """Step 3: Document-level retrieval."""
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
        return doc_avg_scores[:self.config.top_documents]

    def _merge_context_and_document_results(
        self,
        expanded_chunks: List[ChunkInfo],
        top_docs: List[Tuple[DocumentInfo, float]]
    ) -> List[SearchResultWithContext]:
        """Merge context-expanded chunks with document-level results."""
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

        return results

    async def _apply_rerank(
        self,
        query: str,
        results: List[SearchResultWithContext],
    ) -> List[SearchResultWithContext]:
        """Apply Cross-Encoder reranking."""
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

            if self._db.vector_enabled:
                try:
                    texts = [c[0] for c in candidates]
                    embeddings = await self._embedding_provider.embed_batch(texts)
                    candidates = [
                        (c[0], c[1], emb)
                        for c, emb in zip(candidates, embeddings)
                    ]
                except Exception as e:
                    logger.warning("Could not get embeddings for rerank/dedup: {}", e)

            reranked = await self._rerank_service.rerank_and_dedup(query, candidates)

            final_results: List[SearchResultWithContext] = []
            for original_idx, result, new_score in reranked:
                result.final_score = new_score
                final_results.append(result)

            for i, r in enumerate(final_results):
                r.rank = i + 1

            return final_results

        except Exception as e:
            logger.warning("Rerank failed, returning original results: {}", e)
            return results

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
        key_bytes += f":rrfk={self.config.rrf_k}".encode("utf-8")
        key_bytes += f":rerank={self.config.enable_rerank}".encode("utf-8")
        key_bytes += f":rerankt={self.config.rerank_threshold}".encode("utf-8")
        key_bytes += f":dedupt={self.config.dedup_threshold}".encode("utf-8")
        key_bytes += f":ctxexpand={self.config.enable_context_expansion}".encode("utf-8")
        key_bytes += f":doclevel={self.config.enable_document_level}".encode("utf-8")
        return hashlib.sha256(key_bytes).hexdigest()
