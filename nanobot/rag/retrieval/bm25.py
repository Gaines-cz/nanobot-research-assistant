"""BM25 full-text search retriever.

This module provides BM25-based full-text search using SQLite FTS5.
"""

import time
from typing import List

from loguru import logger

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
        """Full-text search using FTS5 with hybrid strategy: phrase + OR.

        Strategy:
        1. First try exact phrase match for high precision
        2. Then try OR keyword match for high recall
        3. Merge results, phrase matches first, deduplicated
        4. Add neighbor chunks (prev/next) for better recall
        """
        search_start = time.perf_counter()

        # Generate both query types
        phrase_query = self._sanitize_fts_query(query)
        or_query = self._sanitize_fts_query_or(query)

        # If both queries are empty, return fallback results
        if not phrase_query and not or_query:
            return self._get_fallback_results(top_k)

        # Execute both queries (request more results for merging)
        min_fetch_k = 20
        fetch_k = max(top_k * 2, min_fetch_k)

        phrase_start = time.perf_counter()
        phrase_results = self._query_with_safe_query(phrase_query, fetch_k) if phrase_query else []
        phrase_elapsed = (time.perf_counter() - phrase_start) * 1000

        or_start = time.perf_counter()
        or_results = self._query_with_safe_query(or_query, fetch_k) if or_query else []
        or_elapsed = (time.perf_counter() - or_start) * 1000

        # Log the results for debugging
        logger.debug("[RAG] Phrase search: {} results, OR search: {} results",
                     len(phrase_results), len(or_results))
        logger.debug("[RAG PERF] BM25 phrase={:.1f}ms, OR={:.1f}ms",
                     phrase_elapsed, or_elapsed)

        # Merge results: phrase first, then OR, deduplicated
        merge_start = time.perf_counter()
        seen = set()
        merged: List[SearchResult] = []

        for r in phrase_results:
            key = f"{r.path}:{r.chunk_index}"
            if key not in seen:
                seen.add(key)
                merged.append(r)

        for r in or_results:
            key = f"{r.path}:{r.chunk_index}"
            if key not in seen:
                seen.add(key)
                merged.append(r)

        # Neighbor chunk expansion is disabled to avoid position-score mismatch in percentile fusion
        # If context is needed, use Step2 context_expansion instead
        # if merged:
        #     neighbor_start = time.perf_counter()
        #     neighbor_chunks = self._fetch_neighbor_chunks(merged)
        #     neighbor_elapsed = (time.perf_counter() - neighbor_start) * 1000
        #     logger.debug("[RAG PERF] BM25 neighbor fetch: {} chunks, elapsed={:.1f}ms",
        #                  len(neighbor_chunks), neighbor_elapsed)
        #     for r in neighbor_chunks:
        #         key = f"{r.path}:{r.chunk_index}"
        #         if key not in seen:
        #             seen.add(key)
        #             merged.append(r)

        merge_elapsed = (time.perf_counter() - merge_start) * 1000
        logger.debug("[RAG PERF] BM25 merge: elapsed={:.1f}ms", merge_elapsed)

        # Truncate to top_k
        final_results = merged[:top_k]

        total_elapsed = (time.perf_counter() - search_start) * 1000
        logger.info("[RAG PERF] BM25 full-text: {} results, elapsed={:.1f}ms",
                    len(final_results), total_elapsed)

        return final_results

    def _fetch_neighbor_chunks(self, base_results: List[SearchResult]) -> List[SearchResult]:
        """Fetch neighboring chunks (prev and next) for each result.

        Returns a list of neighbor chunks with lower scores.
        """
        if not base_results:
            return []

        db = self._db.db
        neighbors: List[SearchResult] = []

        for result in base_results:
            # Fetch prev chunk (index - 1) and next chunk (index + 1)
            for offset in (-1, 1):
                try:
                    if self.config.enable_dual_granularity:
                        cursor = db.execute("""
                            SELECT
                                d.path,
                                d.filename,
                                c.chunk_index,
                                c.content
                            FROM chunks c
                            JOIN documents d ON c.doc_id = d.id
                            WHERE d.path = ?
                              AND c.chunk_index = ?
                              AND (c.granularity = 'large' OR c.granularity IS NULL)
                        """, (result.path, result.chunk_index + offset))
                    else:
                        cursor = db.execute("""
                            SELECT
                                d.path,
                                d.filename,
                                c.chunk_index,
                                c.content
                            FROM chunks c
                            JOIN documents d ON c.doc_id = d.id
                            WHERE d.path = ?
                              AND c.chunk_index = ?
                        """, (result.path, result.chunk_index + offset))

                    row = cursor.fetchone()
                    if row:
                        # Give neighbor a lower score than original result
                        neighbor_score = result.score * 0.8
                        neighbors.append(SearchResult(
                            path=row[0],
                            filename=row[1],
                            chunk_index=row[2],
                            content=row[3],
                            score=neighbor_score,
                            source="fulltext_neighbor",
                        ))
                except Exception as e:
                    logger.debug("[RAG] Failed to fetch neighbor chunk: {}", e)
                    continue

        return neighbors

    def _query_with_safe_query(self, safe_query: str, top_k: int) -> List[SearchResult]:
        """Execute FTS query with the given safe query string."""
        db = self._db.db
        results: List[SearchResult] = []

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
            logger.warning("[RAG] FTS query failed: {}", e)

        return results

    def _get_fallback_results(self, top_k: int) -> List[SearchResult]:
        """Get fallback results when query is empty."""
        db = self._db.db
        results: List[SearchResult] = []

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
