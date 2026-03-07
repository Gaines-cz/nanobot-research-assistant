"""Vector similarity search retriever.

This module provides vector-based similarity search using sqlite-vec.
"""

from typing import List

from loguru import logger

from nanobot.rag.models import SearchResult
from nanobot.rag.retrieval.base import Retriever


class VectorRetriever(Retriever):
    """
    Vector similarity retriever using sqlite-vec.

    Supports dual-granularity search where small chunks are searched
    and results are mapped back to their parent large chunks.
    """

    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Perform vector similarity search.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of SearchResult sorted by similarity
        """
        if not self._db.vector_enabled:
            return []

        db = self._db.db
        results: List[SearchResult] = []

        try:
            query_embedding = await self._embedding_provider.embed(query)
            import sqlite_vec
            embedding_blob = sqlite_vec.serialize_float32(query_embedding)

            if self.config.enable_dual_granularity:
                # Dual granularity: search small chunks, map to parent large chunks
                cursor = db.execute("""
                    SELECT
                        d.path,
                        d.filename,
                        c.chunk_index,
                        c.content,
                        c.parent_chunk_id,
                        parent.id as parent_id,
                        parent.chunk_index as parent_index,
                        parent.content as parent_content,
                        e.distance
                    FROM chunk_embeddings e
                    JOIN chunks c ON e.chunk_id = c.id
                    JOIN documents d ON c.doc_id = d.id
                    LEFT JOIN chunks parent ON c.parent_chunk_id = parent.id
                    WHERE c.granularity = 'small'
                      AND e.embedding MATCH ?
                      AND e.k = ?
                """, (embedding_blob, top_k))

                for row in cursor:
                    distance = row[8] if row[8] is not None else 1.0
                    similarity = max(0.0, min(1.0, 1.0 - (distance / 2.0)))

                    # If parent exists, use parent's content and index
                    content = row[7] if row[7] is not None else row[3]
                    chunk_index = row[6] if row[6] is not None else row[2]

                    results.append(SearchResult(
                        path=row[0],
                        filename=row[1],
                        chunk_index=chunk_index,
                        content=content,
                        score=similarity,
                        source="vector",
                    ))
            else:
                # Legacy single granularity
                cursor = db.execute("""
                    SELECT
                        d.path,
                        d.filename,
                        c.chunk_index,
                        c.content,
                        e.distance
                    FROM chunk_embeddings e
                    JOIN chunks c ON e.chunk_id = c.id
                    JOIN documents d ON c.doc_id = d.id
                    WHERE e.embedding MATCH ?
                      AND e.k = ?
                """, (embedding_blob, top_k))

                for row in cursor:
                    distance = row[4] if row[4] is not None else 1.0
                    similarity = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
                    results.append(SearchResult(
                        path=row[0],
                        filename=row[1],
                        chunk_index=row[2],
                        content=row[3],
                        score=similarity,
                        source="vector",
                    ))
        except Exception as e:
            logger.warning("[DualGranularity] Vector search failed: {}", e)
            self._db.record_vector_disabled()
            return []

        return results
