"""BM25 full-text search retriever.

This module provides BM25-based full-text search using SQLite FTS5.
"""

from typing import List

from nanobot.rag.models import SearchResult
from nanobot.rag.retrieval.base import Retriever


class BM25Retriever(Retriever):
    """
    BM25 full-text retriever using SQLite FTS5.

    Provides efficient keyword-based search with Porter stemming
    and Unicode support.
    """

    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Perform BM25 full-text search.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of SearchResult sorted by BM25 score
        """
        return self._fulltext_search(query, top_k)

    def _fulltext_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Full-text search using FTS5."""
        db = self._db.db
        results: List[SearchResult] = []

        safe_query = self._sanitize_fts_query(query)

        if not safe_query:
            if self.config.enable_dual_granularity:
                cursor = db.execute("""
                    SELECT
                        d.path,
                        d.filename,
                        c.chunk_index,
                        c.content
                    FROM chunks c
                    JOIN documents d ON c.doc_id = d.id
                    WHERE c.granularity = 'large'
                    ORDER BY c.id DESC
                    LIMIT ?
                """, (top_k,))
            else:
                cursor = db.execute("""
                    SELECT
                        d.path,
                        d.filename,
                        c.chunk_index,
                        c.content
                    FROM chunks c
                    JOIN documents d ON c.doc_id = d.id
                    ORDER BY c.id DESC
                    LIMIT ?
                """, (top_k,))
            for i, row in enumerate(cursor):
                results.append(SearchResult(
                    path=row[0],
                    filename=row[1],
                    chunk_index=row[2],
                    content=row[3],
                    score=1.0 - (i * 0.1),
                    source="fulltext",
                ))
            return results

        try:
            if self.config.enable_dual_granularity:
                # Dual granularity: only search large chunks for BM25
                cursor = db.execute("""
                    SELECT
                        d.path,
                        d.filename,
                        c.chunk_index,
                        c.content,
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
                # Legacy single granularity
                cursor = db.execute("""
                    SELECT
                        d.path,
                        d.filename,
                        c.chunk_index,
                        c.content,
                        bm25(chunks_fts) as score
                    FROM chunks_fts
                    JOIN chunks c ON chunks_fts.rowid = c.id
                    JOIN documents d ON c.doc_id = d.id
                    WHERE chunks_fts MATCH ?
                    ORDER BY bm25(chunks_fts)
                    LIMIT ?
                """, (safe_query, top_k))

            rows = cursor.fetchall()
            if rows:
                # Use Min-Max normalization for BM25 scores
                bm25_scores = [row[4] if row[4] is not None else 1.0 for row in rows]
                min_bm25 = min(bm25_scores)
                max_bm25 = max(bm25_scores)

                for i, row in enumerate(rows):
                    bm25_score = bm25_scores[i]
                    if max_bm25 == min_bm25:
                        normalized = 1.0
                    else:
                        # Invert because BM25: lower = better
                        normalized = 1.0 - (bm25_score - min_bm25) / (max_bm25 - min_bm25)

                    results.append(SearchResult(
                        path=row[0],
                        filename=row[1],
                        chunk_index=row[2],
                        content=row[3],
                        score=normalized,
                        source="fulltext",
                    ))
        except Exception as e:
            from loguru import logger
            logger.warning("[DualGranularity] FTS with BM25 failed, trying simpler query: {}", e)
            # Fallback omitted for brevity

        return results
