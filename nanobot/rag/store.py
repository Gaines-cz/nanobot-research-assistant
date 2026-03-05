"""Document store facade - orchestrates connection, indexing, and search."""

from pathlib import Path
from typing import Any, Optional

from nanobot.config.schema import RAGConfig
from nanobot.rag.connection import DatabaseConnection
from nanobot.rag.embeddings import EmbeddingProvider
from nanobot.rag.indexer import DocumentIndexer
from nanobot.rag.search import (
    DocumentSearch,
    SearchResult,
    SearchResultWithContext,
)


class DocumentStore:
    """
    Unified document store facade.

    Delegates to:
    - DatabaseConnection: Connection management and schema
    - DocumentIndexer: Document indexing operations
    - DocumentSearch: Search operations

    Uses:
    - SQLite for metadata storage
    - sqlite-vec for vector embeddings (optional, falls back to FTS only)
    - SQLite FTS5 for full-text search
    """

    def __init__(self, db_path: Path, embedding_provider: Optional[EmbeddingProvider] = None, config: Optional[RAGConfig] = None):
        """
        Initialize document store.

        Args:
            db_path: Path to SQLite database file
            embedding_provider: Optional embedding provider for vector search
            config: RAG configuration
        """
        self.db_path = db_path
        self.embedding_provider = embedding_provider
        self.config = config or RAGConfig()

        # Initialize sub-modules
        self._connection = DatabaseConnection(db_path, embedding_provider, config)
        self._indexer = DocumentIndexer(self._connection, embedding_provider, config)
        self._search = DocumentSearch(self._connection, embedding_provider, config)

    # === Indexing Operations ===

    async def scan_and_index(
        self,
        docs_dir: Path,
        chunk_size: Optional[int] = None,
        chunk_overlap_ratio: Optional[float] = None,
        root_path: Optional[Path] = None,
    ) -> dict[str, int]:
        """
        Scan docs_dir and index documents.

        Args:
            docs_dir: Directory to scan
            chunk_size: Optional override for max_chunk_size
            chunk_overlap_ratio: Optional override for overlap ratio
            root_path: Optional root path to filter documents for deletion.

        Returns:
            Dict with counts: {"added": n, "updated": n, "deleted": n}
        """
        return await self._indexer.scan_and_index(docs_dir, chunk_size, chunk_overlap_ratio, root_path)

    async def index_single_file(
        self,
        file_path: Path,
        chunk_size: Optional[int] = None,
        chunk_overlap_ratio: Optional[float] = None,
    ) -> bool:
        """
        索引单个文件（增量更新用）。

        Args:
            file_path: 要索引的文件路径
            chunk_size: 可选的分块大小覆盖
            chunk_overlap_ratio: 可选的分块重叠率覆盖

        Returns:
            True: 文件被成功索引（新增或更新）
            False: 文件未变化、无效、或处理失败
        """
        return await self._indexer.index_single_file(
            file_path, chunk_size, chunk_overlap_ratio
        )

    async def schedule_index_update(
        self,
        docs_dir: Path,
        chunk_size: Optional[int] = None,
        chunk_overlap_ratio: Optional[float] = None,
    ) -> None:
        """Schedule an index update after a delay to batch multiple changes."""
        await self._indexer.schedule_index_update(docs_dir, chunk_size, chunk_overlap_ratio)

    # === Search Operations ===

    async def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Hybrid search: vector + full-text with RRF reranking.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of SearchResult sorted by relevance
        """
        return await self._search.search(query, top_k)

    async def search_advanced(self, query: str) -> list[SearchResultWithContext]:
        """
        Advanced multi-step search pipeline.

        1. Core chunk recall (BM25 + vector, dual thresholds, Top5)
        2. Context expansion (prev1 + core + next1)
        3. Document-level prioritization (Top3 docs)
        4. Cross-Encoder rerank + semantic dedup

        Args:
            query: Search query

        Returns:
            List of SearchResultWithContext with expanded context
        """
        return await self._search.search_advanced(query)

    # === Statistics ===

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the document store."""
        db = self._connection.db

        cursor = db.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]

        cursor = db.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = cursor.fetchone()[0]

        cursor = db.execute("SELECT file_type, COUNT(*) FROM documents GROUP BY file_type")
        by_type = {row[0]: row[1] for row in cursor}

        # Average chunk size
        avg_chunk_size = 0
        if chunk_count > 0:
            cursor = db.execute("SELECT AVG(LENGTH(content)) FROM chunks")
            avg_chunk_size = cursor.fetchone()[0] or 0

        # Last scan time
        last_scan_at = None
        if doc_count > 0:
            cursor = db.execute("SELECT MAX(stored_at) FROM documents")
            last_scan_at = cursor.fetchone()[0]

        return {
            "documents": doc_count,
            "chunks": chunk_count,
            "by_file_type": by_type,
            "avg_chunk_size": round(avg_chunk_size, 2),
            "last_scan_at": last_scan_at,
            "vector_enabled": self._connection.vector_enabled,
        }

    def is_vector_enabled(self) -> bool:
        """Check if vector search is enabled."""
        return self._connection.vector_enabled

    # === Connection Management ===

    def close(self) -> None:
        """Close the database connection."""
        self._connection.close()

    def clear_cache(self) -> None:
        """Clear search cache."""
        self._search.clear_cache()
