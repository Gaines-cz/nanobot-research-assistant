"""Indexing pipeline for documents.

This module provides the document indexing pipeline that coordinates
parsing, chunking, and storing documents.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

from nanobot.config.schema import RAGConfig
from nanobot.rag.embeddings import EmbeddingProvider
from nanobot.rag.indexing.chunker import (
    HierarchicalChunker,
    HierarchicalChunks,
    SemanticChunk,
)
from nanobot.rag.indexing.parser import DocumentParser
from nanobot.rag.models import Document
from nanobot.rag.storage.connection import DatabaseConnection


class IndexingPipeline:
    """
    Pipeline for indexing documents.

    Coordinates:
    1. Document parsing
    2. Text chunking
    3. Embedding generation
    4. Database storage
    """

    def __init__(
        self,
        db_connection: DatabaseConnection,
        embedding_provider: Optional[EmbeddingProvider] = None,
        config: Optional[RAGConfig] = None,
    ):
        """
        Initialize indexing pipeline.

        Args:
            db_connection: Database connection manager
            embedding_provider: Optional embedding provider
            config: RAG configuration
        """
        self._db = db_connection
        self._embedding_provider = embedding_provider
        self.config = config or RAGConfig()

        self._parser = DocumentParser()
        self._chunker = HierarchicalChunker(
            small_min=config.small_min_chunk_size,
            small_max=config.small_max_chunk_size,
            small_overlap=config.small_chunk_overlap_ratio,
            large_min=config.large_min_chunk_size,
            large_max=config.large_max_chunk_size,
            large_overlap=config.large_chunk_overlap_ratio,
        )

    def parse_document(self, path: Path) -> Tuple[str, str, Dict]:
        """
        Parse a document and extract its content and metadata.

        Args:
            path: Path to the document

        Returns:
            Tuple of (content, file_type, metadata)
        """
        content, file_type = self._parser.parse(path)
        metadata = self._parser.extract_metadata(path, content)
        return content, file_type, metadata

    def chunk_document(
        self,
        content: str,
        min_chunk_size: Optional[int] = None,
        max_chunk_size: Optional[int] = None,
        chunk_overlap_ratio: Optional[float] = None,
    ) -> HierarchicalChunks:
        """
        Chunk document content into hierarchical chunks.

        Args:
            content: Document content to chunk
            min_chunk_size: Optional override for min chunk size
            max_chunk_size: Optional override for max chunk size
            chunk_overlap_ratio: Optional override for overlap ratio

        Returns:
            HierarchicalChunks with large and small chunks
        """
        # Use overrides if provided, otherwise use config
        if min_chunk_size is not None and max_chunk_size is not None and chunk_overlap_ratio is not None:
            # Create a temporary chunker with overridden parameters
            chunker = HierarchicalChunker(
                small_min=min_chunk_size // 2,
                small_max=min_chunk_size,
                small_overlap=chunk_overlap_ratio,
                large_min=min_chunk_size,
                large_max=max_chunk_size,
                large_overlap=chunk_overlap_ratio,
            )
            return chunker.chunk_hierarchical(content)
        else:
            # Use configured chunker
            return self._chunker.chunk_hierarchical(content)

    async def index_document(
        self,
        path: Path,
        min_chunk_size: Optional[int] = None,
        max_chunk_size: Optional[int] = None,
        chunk_overlap_ratio: Optional[float] = None,
    ) -> Optional[Document]:
        """
        Process and index a single document.

        Args:
            path: Path to the document
            min_chunk_size: Optional override for min chunk size
            max_chunk_size: Optional override for max chunk size
            chunk_overlap_ratio: Optional override for overlap ratio

        Returns:
            Document object if indexed, None if skipped
        """
        # Get file metadata
        stat = path.stat()
        mtime = stat.st_mtime
        file_size = stat.st_size

        # Check if document already exists and is up to date
        db = self._db.db
        cursor = db.execute("""
            SELECT id, mtime FROM documents WHERE path = ?
        """, (str(path),))
        row = cursor.fetchone()

        if row:
            doc_id, stored_mtime = row
            if abs(stored_mtime - mtime) < 1:
                logger.debug("Document unchanged, skipping: {}", path)
                return None
            # Delete old version
            await self.delete_document(doc_id)

        # Parse document
        content, file_type, metadata = self.parse_document(path)

        # Chunk document
        hierarchical_chunks = self.chunk_document(
            content,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            chunk_overlap_ratio=chunk_overlap_ratio,
        )

        if not hierarchical_chunks.large_chunks and not hierarchical_chunks.small_chunks:
            logger.warning("No chunks generated for document: {}", path)
            return None

        # Insert document
        stored_at = time.time()
        cursor = db.execute("""
            INSERT INTO documents
            (path, filename, file_type, file_size, mtime, stored_at, title, doc_type, abstract)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(path),
            path.name,
            file_type,
            file_size,
            mtime,
            stored_at,
            metadata.get("title"),
            metadata.get("doc_type"),
            metadata.get("abstract"),
        ))
        doc_id = cursor.lastrowid

        # Insert chunks
        chunk_map: Dict[int, int] = {}  # large_chunk_list_idx -> db_chunk_id

        # First insert large chunks
        for i, large_chunk in enumerate(hierarchical_chunks.large_chunks):
            cursor = db.execute("""
                INSERT INTO chunks
                (doc_id, chunk_index, content, start_pos, end_pos, chunk_type, section_title, granularity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc_id,
                i,
                large_chunk.content,
                large_chunk.start_pos,
                large_chunk.end_pos,
                large_chunk.chunk_type,
                large_chunk.section_title,
                "large",
            ))
            chunk_map[i] = cursor.lastrowid

        # Then insert small chunks with parent references
        for i, small_chunk in enumerate(hierarchical_chunks.small_chunks):
            large_idx = hierarchical_chunks.small_to_large.get(i)
            parent_chunk_id = chunk_map.get(large_idx) if large_idx is not None else None

            cursor = db.execute("""
                INSERT INTO chunks
                (doc_id, chunk_index, content, start_pos, end_pos, chunk_type, section_title, granularity, parent_chunk_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc_id,
                i,
                small_chunk.content,
                small_chunk.start_pos,
                small_chunk.end_pos,
                small_chunk.chunk_type,
                small_chunk.section_title,
                "small",
                parent_chunk_id,
            ))

        # Generate embeddings if enabled
        # In dual granularity mode: generate embeddings for both small and large chunks
        # - small chunks: used for vector retrieval
        # - large chunks: used for evaluation and similarity calculation
        if self.config.enable_dual_granularity:
            await self._generate_embeddings(doc_id, hierarchical_chunks.small_chunks, granularity_label="small")
            await self._generate_embeddings(doc_id, hierarchical_chunks.large_chunks, granularity_label="large")
        else:
            # In single granularity mode, generate embeddings for large chunks
            await self._generate_embeddings(doc_id, hierarchical_chunks.large_chunks)

        db.commit()
        logger.info("Indexed document: {} ({} chunks)", path, len(hierarchical_chunks.large_chunks))

        # Return the document
        return Document(
            id=doc_id,
            path=str(path),
            filename=path.name,
            file_type=file_type,
            file_size=file_size,
            mtime=mtime,
            stored_at=stored_at,
            title=metadata.get("title"),
            doc_type=metadata.get("doc_type"),
            abstract=metadata.get("abstract"),
        )

    async def delete_document(self, doc_id: int) -> bool:
        """
        Delete a document and all its chunks.

        Args:
            doc_id: Document ID to delete

        Returns:
            True if deleted, False otherwise
        """
        db = self._db.db
        db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        db.commit()
        return True

    async def delete_document_by_path(self, path: Path) -> bool:
        """
        Delete a document by path.

        Args:
            path: Path to the document

        Returns:
            True if deleted, False otherwise
        """
        db = self._db.db
        cursor = db.execute("SELECT id FROM documents WHERE path = ?", (str(path),))
        row = cursor.fetchone()
        if row:
            await self.delete_document(row[0])
            return True
        return False

    async def _generate_embeddings(
        self,
        doc_id: int,
        chunks: List[SemanticChunk],
        granularity_label: Optional[str] = None,
    ) -> None:
        """Generate and store embeddings for chunks.

        Args:
            doc_id: Document ID
            chunks: List of chunks to embed
            granularity_label: Optional granularity label ('small' or 'large')
        """
        if not self._embedding_provider or not self._db.vector_enabled:
            return

        db = self._db.db

        # Get chunk IDs for this doc, filtered by granularity if specified
        granularity_filter = "AND granularity = ?" if granularity_label else ""
        params = (doc_id, granularity_label) if granularity_label else (doc_id,)

        cursor = db.execute(f"""
            SELECT id, chunk_index FROM chunks
            WHERE doc_id = ? {granularity_filter}
            ORDER BY chunk_index
        """, params)
        chunk_id_map = {row[1]: row[0] for row in cursor.fetchall()}

        # Generate embeddings
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = await self._embedding_provider.embed_batch(chunk_texts)

        import sqlite_vec

        # Store embeddings
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = chunk_id_map.get(i)
            if chunk_id is not None:
                embedding_blob = sqlite_vec.serialize_float32(embedding)
                db.execute("""
                    INSERT OR REPLACE INTO chunk_embeddings (chunk_id, embedding)
                    VALUES (?, ?)
                """, (chunk_id, embedding_blob))
