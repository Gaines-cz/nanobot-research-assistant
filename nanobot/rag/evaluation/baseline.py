"""RAG Evaluation - Baseline retrievers for comparison."""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from loguru import logger

from nanobot.rag.storage.connection import DatabaseConnection


@dataclass
class BaselineChunkInfo:
    """Minimal chunk info for baseline evaluation."""
    id: int
    doc_id: int
    chunk_index: int
    content: str
    embedding: Optional[List[float]] = None


@dataclass
class BaselineDocumentInfo:
    """Minimal document info for baseline evaluation."""
    id: int
    path: str
    filename: str


@dataclass
class BaselineSearchResult:
    """Minimal search result for baseline evaluation (compatible with judge)."""
    document: BaselineDocumentInfo
    chunk: BaselineChunkInfo


class BaselineRetriever:
    """
    Baseline retrievers for comparison.

    Provides:
    - BM25-only search (no vector)
    - Keyword-only search (simple LIKE)
    """

    def __init__(self, db_connection: DatabaseConnection):
        self.db_connection = db_connection
        # Check if dual granularity is enabled by looking at schema
        self._use_dual_granularity = self._check_dual_granularity()
        # Check if vector embeddings are available
        self._has_embeddings = self._check_embeddings_table()

    def _check_dual_granularity(self) -> bool:
        """Check if database has granularity column."""
        try:
            cursor = self.db_connection.db.execute("""
                PRAGMA table_info(chunks)
            """)
            columns = [row[1] for row in cursor.fetchall()]
            return 'granularity' in columns
        except Exception:
            return False

    def _check_embeddings_table(self) -> bool:
        """Check if chunk_embeddings table exists."""
        try:
            cursor = self.db_connection.db.execute("""
                SELECT name FROM sqlite_master WHERE type='table' AND name='chunk_embeddings'
            """)
            return cursor.fetchone() is not None
        except Exception:
            return False

    def _load_embedding_for_chunk(self, chunk_id: int) -> Optional[List[float]]:
        """Load embedding for a chunk from database."""
        if not self._has_embeddings:
            return None

        db = self.db_connection.db
        try:
            import sqlite_vec
            cursor = db.execute("""
                SELECT embedding FROM chunk_embeddings WHERE chunk_id = ?
            """, (chunk_id,))
            row = cursor.fetchone()
            if row and row[0]:
                # Deserialize the float32 embedding
                embedding_blob = row[0]
                return list(sqlite_vec.deserialize_float32(embedding_blob))
        except Exception:
            # Silently fail - embedding is optional
            pass
        return None

    async def search_bm25(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[BaselineSearchResult]:
        """
        BM25-only search (baseline).

        Uses only FTS5, no vector search.
        Returns complete results with embeddings for judge compatibility.
        """
        db = self.db_connection.db
        results: List[BaselineSearchResult] = []

        try:
            # Sanitize query for FTS
            safe_query = self._sanitize_fts_query(query)

            if not safe_query:
                # Fallback: return recent chunks
                if self._use_dual_granularity:
                    cursor = db.execute("""
                        SELECT c.id, c.doc_id, c.chunk_index, c.content, d.path, d.filename
                        FROM chunks c
                        JOIN documents d ON c.doc_id = d.id
                        WHERE c.granularity = 'large'
                        ORDER BY c.id DESC
                        LIMIT ?
                    """, (top_k,))
                else:
                    cursor = db.execute("""
                        SELECT c.id, c.doc_id, c.chunk_index, c.content, d.path, d.filename
                        FROM chunks c
                        JOIN documents d ON c.doc_id = d.id
                        ORDER BY c.id DESC
                        LIMIT ?
                    """, (top_k,))
                for i, row in enumerate(cursor):
                    chunk_id, doc_id, chunk_idx, content, path, filename = row
                    embedding = self._load_embedding_for_chunk(chunk_id)
                    doc_info = BaselineDocumentInfo(id=doc_id, path=path, filename=filename)
                    chunk_info = BaselineChunkInfo(
                        id=chunk_id, doc_id=doc_id, chunk_index=chunk_idx,
                        content=content, embedding=embedding
                    )
                    results.append(BaselineSearchResult(document=doc_info, chunk=chunk_info))
                return results

            if self._use_dual_granularity:
                cursor = db.execute("""
                    SELECT
                        c.id, c.doc_id, c.chunk_index, c.content,
                        d.path, d.filename,
                        bm25(chunks_fts) as score
                    FROM chunks_fts
                    JOIN chunks c ON chunks_fts.rowid = c.id
                    JOIN documents d ON c.doc_id = d.id
                    WHERE c.granularity = 'large'
                      AND chunks_fts MATCH ?
                    ORDER BY bm25(chunks_fts)
                    LIMIT ?
                """, (safe_query, top_k))
            else:
                cursor = db.execute("""
                    SELECT
                        c.id, c.doc_id, c.chunk_index, c.content,
                        d.path, d.filename,
                        bm25(chunks_fts) as score
                    FROM chunks_fts
                    JOIN chunks c ON chunks_fts.rowid = c.id
                    JOIN documents d ON c.doc_id = d.id
                    WHERE chunks_fts MATCH ?
                    ORDER BY bm25(chunks_fts)
                    LIMIT ?
                """, (safe_query, top_k))

            for row in cursor:
                chunk_id, doc_id, chunk_idx, content, path, filename, _ = row
                embedding = self._load_embedding_for_chunk(chunk_id)
                doc_info = BaselineDocumentInfo(id=doc_id, path=path, filename=filename)
                chunk_info = BaselineChunkInfo(
                    id=chunk_id, doc_id=doc_id, chunk_index=chunk_idx,
                    content=content, embedding=embedding
                )
                results.append(BaselineSearchResult(document=doc_info, chunk=chunk_info))

        except Exception as e:
            logger.warning("BM25 search failed for query '{}': {}", query, e)
            # Fallback to empty results
            pass

        return results

    async def search_keyword(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[int, float, str]]:
        """
        Keyword-only search (simplest baseline).

        Uses simple LIKE queries.
        """
        db = self.db_connection.db
        results: List[Tuple[int, float, str]] = []

        try:
            # Simple LIKE search
            search_pattern = f"%{query}%"
            cursor = db.execute("""
                SELECT id, content
                FROM chunks
                WHERE content LIKE ?
                LIMIT ?
            """, (search_pattern, top_k))

            for i, row in enumerate(cursor):
                score = 1.0 - (i * 0.1)
                results.append((row[0], score, row[1]))

        except Exception as e:
            logger.warning("Keyword search failed for query '{}': {}", query, e)
            pass

        return results

    def _sanitize_fts_query(self, query: str) -> str:
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
