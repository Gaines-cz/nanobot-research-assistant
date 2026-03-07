"""Base classes for retrievers.

This module provides the foundation for all retriever implementations,
extracted from search.py.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Optional

from nanobot.config.schema import RAGConfig
from nanobot.rag.cache import SearchCacheManager
from nanobot.rag.embeddings import EmbeddingProvider
from nanobot.rag.models import (
    SearchResult,
    SearchResultWithContext,
)
from nanobot.rag.query import QueryExpander
from nanobot.rag.storage.connection import DatabaseConnection


class RetrievalConstants:
    """检索相关的常量定义。"""

    # 召回阶段检索数量 - 更多候选 = 更好的 recall
    RECALL_TOP_K = 50

    # 最小结果阈值 - 过滤后结果少于此值时放宽阈值
    MIN_RESULTS_THRESHOLD = 3

    # 默认返回结果数量
    DEFAULT_TOP_K = 5


class BaseRetriever(ABC):
    """Base class for all retrievers with common functionality."""

    def __init__(
        self,
        db_connection: DatabaseConnection,
        embedding_provider: Optional[EmbeddingProvider] = None,
        config: Optional[RAGConfig] = None,
    ):
        """
        Initialize base retriever.

        Args:
            db_connection: Database connection manager
            embedding_provider: Optional embedding provider
            config: RAG configuration
        """
        self._db = db_connection
        self._embedding_provider = embedding_provider
        self.config = config or RAGConfig()

        # Initialize cache manager
        self._cache_manager = SearchCacheManager(
            max_size=1000,
            ttl_seconds=self.config.cache_ttl_seconds
        )

        # Initialize query expander
        self._query_expander = QueryExpander(enabled=self.config.enable_query_expand)

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Sanitize query for FTS5 simple phrase search.

        This is a robust approach that avoids FTS5 query syntax conflicts:
        1. Replace all punctuation/special chars with spaces (including hyphens)
        2. Normalize whitespace
        3. Return as a single quoted phrase for safe literal matching

        Why? We don't need FTS5's advanced features (col:term, AND/OR, etc.).
        We just want simple keyword matching against the content.
        """
        # Replace anything that's not a letter, number, or Chinese with space
        # This includes hyphens, which prevents "mini-max" being parsed as "column:term"
        sanitized = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff]', ' ', query)
        # Normalize multiple spaces to single space
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()

        if not sanitized:
            return ""

        # Return as a single quoted phrase - this is the safest way to avoid
        # ANY FTS5 query syntax interpretation (keywords, operators, etc.)
        return f'"{sanitized}"'

    def clear_cache(self) -> None:
        """Clear search cache."""
        self._cache_manager.clear_all()


class Retriever(BaseRetriever, ABC):
    """Abstract base class for basic retrievers."""

    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Search for relevant chunks.

        Args:
            query: The search query string
            top_k: Maximum number of results to return

        Returns:
            List of SearchResult objects sorted by relevance
        """
        pass


class AdvancedRetriever(Retriever, ABC):
    """Abstract base class for advanced retrievers with context expansion."""

    @abstractmethod
    async def search_advanced(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[SearchResultWithContext]:
        """
        Advanced search with context expansion and reranking.

        Args:
            query: The search query string
            top_k: Maximum number of results to return (optional)

        Returns:
            List of SearchResultWithContext objects with expanded context
        """
        pass
