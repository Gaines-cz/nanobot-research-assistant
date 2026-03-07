"""Context expansion for search results.

This module provides functionality to expand search results with
surrounding context chunks.
"""

from typing import List, Optional

from nanobot.config.schema import RAGConfig
from nanobot.rag.models import ChunkInfo, SearchResult
from nanobot.rag.storage.connection import DatabaseConnection


class ContextExpander:
    """
    Expands search results with surrounding context.

    Adds previous and next chunks to each search result to provide
    more complete context for the LLM.
    """

    def __init__(
        self,
        db_connection: DatabaseConnection,
        config: Optional[RAGConfig] = None,
    ):
        """
        Initialize context expander.

        Args:
            db_connection: Database connection manager
            config: RAG configuration
        """
        self._db = db_connection
        self.config = config or RAGConfig()

    def expand_context(
        self,
        core_results: List[SearchResult],
        enable_context_expansion: bool = True
    ) -> List[ChunkInfo]:
        """
        Expand context around core chunks.

        Args:
            core_results: List of core search results
            enable_context_expansion: Whether to enable context expansion

        Returns:
            List of ChunkInfo objects with expanded context
        """
        if not enable_context_expansion:
            chunks = []
            db = self._db.db
            for result in core_results:
                # Prefer large chunk in dual granularity mode
                if self.config.enable_dual_granularity:
                    cursor = db.execute("""
                        SELECT c.id, c.doc_id, c.chunk_index, c.content, c.chunk_type, c.section_title
                        FROM chunks c
                        JOIN documents d ON c.doc_id = d.id
                        WHERE d.path = ? AND c.chunk_index = ?
                          AND (c.granularity = 'large' OR c.granularity IS NULL)
                    """, (result.path, result.chunk_index))
                else:
                    cursor = db.execute("""
                        SELECT c.id, c.doc_id, c.chunk_index, c.content, c.chunk_type, c.section_title
                        FROM chunks c
                        JOIN documents d ON c.doc_id = d.id
                        WHERE d.path = ? AND c.chunk_index = ?
                    """, (result.path, result.chunk_index))
                row = cursor.fetchone()
                if row:
                    # Load embedding for this chunk
                    embedding = self._load_embedding_for_chunk(row[0])
                    chunks.append(ChunkInfo(
                        id=row[0],
                        doc_id=row[1],
                        chunk_index=row[2],
                        content=row[3],
                        score=result.score,
                        source=result.source,
                        chunk_type=row[4],
                        section_title=row[5],
                        embedding=embedding,
                    ))
            return chunks

        db = self._db.db
        expanded_chunks = []
        prev_count = self.config.context_prev_chunks
        next_count = self.config.context_next_chunks

        for result in core_results:
            # Prefer large chunk in dual granularity mode
            if self.config.enable_dual_granularity:
                cursor = db.execute("""
                    SELECT c.id, c.doc_id, c.chunk_index, c.content, c.chunk_type, c.section_title
                    FROM chunks c
                    JOIN documents d ON c.doc_id = d.id
                    WHERE d.path = ? AND c.chunk_index = ?
                      AND (c.granularity = 'large' OR c.granularity IS NULL)
                """, (result.path, result.chunk_index))
            else:
                cursor = db.execute("""
                    SELECT c.id, c.doc_id, c.chunk_index, c.content, c.chunk_type, c.section_title
                    FROM chunks c
                    JOIN documents d ON c.doc_id = d.id
                    WHERE d.path = ? AND c.chunk_index = ?
                """, (result.path, result.chunk_index))
            row = cursor.fetchone()
            if not row:
                continue

            chunk_id, doc_id, chunk_idx, content, chunk_type, section_title = row

            # Load embedding for this chunk
            embedding = self._load_embedding_for_chunk(chunk_id)

            # Get prev chunks - filter by large granularity in dual mode
            prev_contents = []
            for i in range(1, prev_count + 1):
                if self.config.enable_dual_granularity:
                    cursor = db.execute("""
                        SELECT content FROM chunks
                        WHERE doc_id = ? AND chunk_index = ? AND granularity = 'large'
                    """, (doc_id, chunk_idx - i))
                else:
                    cursor = db.execute("""
                        SELECT content FROM chunks
                        WHERE doc_id = ? AND chunk_index = ?
                    """, (doc_id, chunk_idx - i))
                prev_row = cursor.fetchone()
                if prev_row:
                    prev_contents.insert(0, prev_row[0])
            prev_content = "\n\n".join(prev_contents) if prev_contents else None

            # Get next chunks - filter by large granularity in dual mode
            next_contents = []
            for i in range(1, next_count + 1):
                if self.config.enable_dual_granularity:
                    cursor = db.execute("""
                        SELECT content FROM chunks
                        WHERE doc_id = ? AND chunk_index = ? AND granularity = 'large'
                    """, (doc_id, chunk_idx + i))
                else:
                    cursor = db.execute("""
                        SELECT content FROM chunks
                        WHERE doc_id = ? AND chunk_index = ?
                    """, (doc_id, chunk_idx + i))
                next_row = cursor.fetchone()
                if next_row:
                    next_contents.append(next_row[0])
            next_content = "\n\n".join(next_contents) if next_contents else None

            expanded_chunks.append(ChunkInfo(
                id=chunk_id,
                doc_id=doc_id,
                chunk_index=chunk_idx,
                content=content,
                score=result.score,
                source=result.source,
                chunk_type=chunk_type,
                section_title=section_title,
                prev_content=prev_content,
                next_content=next_content,
                embedding=embedding,
            ))

        return expanded_chunks

    def _load_embedding_for_chunk(self, chunk_id: int) -> Optional[List[float]]:
        """Load embedding for a chunk from database."""
        if not self._db.vector_enabled:
            return None

        db = self._db.db
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
