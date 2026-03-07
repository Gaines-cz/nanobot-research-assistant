"""Hybrid retriever combining vector and BM25 search.

This module provides staged fusion of BM25 and vector search results.
"""

from typing import List, Optional

from loguru import logger

from nanobot.config.schema import RAGConfig
from nanobot.rag.embeddings import EmbeddingProvider
from nanobot.rag.models import SearchResult
from nanobot.rag.retrieval.base import Retriever
from nanobot.rag.retrieval.bm25 import BM25Retriever
from nanobot.rag.retrieval.vector import VectorRetriever
from nanobot.rag.storage.connection import DatabaseConnection


class HybridRetriever(Retriever):
    """
    Hybrid retriever that combines BM25 and vector search.

    Uses a staged fusion approach:
    1. Take BM25 top-k as base results
    2. Add vector results not already in BM25 results
    """

    def __init__(
        self,
        db_connection: DatabaseConnection,
        embedding_provider: Optional[EmbeddingProvider] = None,
        config: Optional[RAGConfig] = None,
    ):
        """
        Initialize hybrid retriever.

        Args:
            db_connection: Database connection manager
            embedding_provider: Optional embedding provider
            config: RAG configuration
        """
        super().__init__(db_connection, embedding_provider, config)

        # Create component retrievers
        self._vector_retriever = VectorRetriever(db_connection, embedding_provider, config)
        self._bm25_retriever = BM25Retriever(db_connection, embedding_provider, config)

    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Perform hybrid search with staged fusion.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of SearchResult sorted by relevance
        """
        # Check cache first
        cache_key = self._get_cache_key(query, top_k)
        if self.config.enable_search_cache:
            cached_results = self._cache_manager.basic.get(cache_key)
            if cached_results is not None:
                logger.debug("Basic search cache hit for query: {}", query)
                return cached_results

        # Expand query
        expanded_query = self._query_expander.expand(query)
        results: List[SearchResult] = []

        if self._db.vector_enabled:
            try:
                # BM25 search: top 50 results (more for base)
                bm25_results = self._bm25_retriever._fulltext_search(expanded_query, 50)
                # Vector search: top 50 results (for supplement)
                vector_results = await self._vector_retriever.search(expanded_query, 50)

                if not vector_results:
                    for result in bm25_results:
                        result.source = "fulltext"
                    results = bm25_results[:top_k]
                else:
                    logger.info("[StagedFusion] Starting - BM25_results={}, vector_results={}",
                                len(bm25_results), len(vector_results))

                    # Stage 1: Take BM25 top 50 as base
                    seen = set()
                    merged = []
                    for r in bm25_results:
                        key = f"{r.path}:{r.chunk_index}"
                        seen.add(key)
                        r.source = "fulltext"
                        merged.append(r)

                    # Stage 2: Add vector results not in BM25 top 50
                    for r in vector_results:
                        key = f"{r.path}:{r.chunk_index}"
                        if key not in seen:
                            seen.add(key)
                            r.source = "vector"
                            merged.append(r)

                    # Take top_k
                    results = merged[:top_k]

                    final_sources = [r.source for r in results]
                    final_scores = [f"{r.score:.4f}" for r in results]
                    logger.info("[StagedFusion] Complete - final_results={}, sources={}, scores={}",
                                len(results), final_sources, final_scores)
            except Exception as e:
                logger.warning("Hybrid search failed, falling back to full-text search only: {}", e)
                self._db.record_vector_disabled()

        if not results:
            fulltext_results = self._bm25_retriever._fulltext_search(expanded_query, top_k)
            for result in fulltext_results:
                result.source = "fulltext"
            results = fulltext_results

        # Store in cache
        if self.config.enable_search_cache:
            self._cache_manager.basic.set(cache_key, results)

        return results
