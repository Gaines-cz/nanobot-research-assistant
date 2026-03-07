"""Document indexer - simplified facade using the new indexing pipeline.

This module maintains backward compatibility with the old DocumentIndexer
interface while delegating to the new pipeline implementation.
"""

import asyncio
from pathlib import Path
from typing import Optional

from loguru import logger

from nanobot.config.schema import RAGConfig
from nanobot.rag.embeddings import EmbeddingProvider
from nanobot.rag.storage.connection import DatabaseConnection


class DocumentIndexer:
    """
    Handles document indexing operations.

    This is a compatibility wrapper around the new IndexingPipeline.
    Responsibilities:
    - Scanning directories for documents
    - Adding new documents
    - Updating modified documents
    - Deleting removed documents
    - Batch index updates
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf", ".md", ".markdown", ".docx", ".doc", ".txt",
        ".rst", ".py", ".js", ".ts", ".html", ".css"
    }

    def __init__(
        self,
        db_connection: DatabaseConnection,
        embedding_provider: Optional[EmbeddingProvider] = None,
        config: Optional[RAGConfig] = None,
    ):
        """
        Initialize document indexer.

        Args:
            db_connection: Database connection manager
            embedding_provider: Optional embedding provider
            config: RAG configuration
        """
        self._db = db_connection
        self._embedding_provider = embedding_provider
        self.config = config or RAGConfig()

        # Import here to avoid circular imports
        from nanobot.rag.indexing.pipeline import IndexingPipeline
        self._pipeline = IndexingPipeline(db_connection, embedding_provider, config)

        # Batch index update
        self._index_pending: bool = False
        self._index_lock = asyncio.Lock()
        self._index_delay_seconds: float = 30.0

    async def scan_and_index(
        self,
        docs_dir: Path,
        min_chunk_size: Optional[int] = None,
        max_chunk_size: Optional[int] = None,
        chunk_overlap_ratio: Optional[float] = None,
        root_path: Optional[Path] = None,
    ) -> dict[str, int]:
        """
        Scan docs_dir and index documents.

        Args:
            docs_dir: Directory to scan
            min_chunk_size: Optional override for min_chunk_size
            max_chunk_size: Optional override for max_chunk_size
            chunk_overlap_ratio: Optional override for overlap ratio
            root_path: Optional root path to filter documents for deletion.
                       Only documents under this root will be considered for deletion.

        Returns:
            Dict with counts: {"added": n, "updated": n, "deleted": n, "skipped": n}
        """
        if self._index_pending:
            logger.debug("Index update already scheduled, skipping immediate update")
            return {"added": 0, "updated": 0, "deleted": 0, "skipped": 0}

        if not docs_dir.exists():
            docs_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Created docs directory: {}", docs_dir)
            return {"added": 0, "updated": 0, "deleted": 0, "skipped": 0}

        db = self._db.db
        stats = {"added": 0, "updated": 0, "deleted": 0, "skipped": 0}

        # Get known documents from DB, filtered by root_path if provided
        if root_path:
            root_str = str(root_path.resolve())
            cursor = db.execute(
                "SELECT id, path, mtime, stored_at FROM documents WHERE path LIKE ?",
                (f"{root_str}%",)
            )
        else:
            cursor = db.execute("SELECT id, path, mtime, stored_at FROM documents")
        known_docs = {row[1]: {"id": row[0], "mtime": row[2], "stored_at": row[3]} for row in cursor}
        logger.debug("Known docs in DB (filtered by root_path {}): {}", root_path, list(known_docs.keys()))

        # Scan files in docs_dir
        seen_paths: set[str] = set()

        for file_path in docs_dir.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.name.startswith("."):
                continue

            ext = file_path.suffix.lower()
            if ext not in self.SUPPORTED_EXTENSIONS:
                continue

            abs_path = str(file_path.resolve())
            seen_paths.add(abs_path)
            mtime = file_path.stat().st_mtime
            logger.debug("Scanned file: {}, abs_path={}, mtime={}", file_path.name, abs_path, mtime)

            if abs_path not in known_docs:
                # New document
                doc = await self._pipeline.index_document(
                    file_path,
                    min_chunk_size=min_chunk_size,
                    max_chunk_size=max_chunk_size,
                    chunk_overlap_ratio=chunk_overlap_ratio,
                )
                if doc:
                    stats["added"] += 1
                    logger.info("Added document: {}", file_path.name)
                else:
                    stats["skipped"] += 1
            elif mtime > known_docs[abs_path]["stored_at"]:
                # Updated document - delete old and add new
                await self._pipeline.delete_document(known_docs[abs_path]["id"])
                doc = await self._pipeline.index_document(
                    file_path,
                    min_chunk_size=min_chunk_size,
                    max_chunk_size=max_chunk_size,
                    chunk_overlap_ratio=chunk_overlap_ratio,
                )
                if doc:
                    stats["updated"] += 1
                    logger.info("Updated document: {}", file_path.name)
                else:
                    stats["skipped"] += 1

        # Delete documents that no longer exist
        logger.debug("Seen paths during scan: {}", seen_paths)
        for path, info in known_docs.items():
            if path not in seen_paths:
                logger.debug("Deleting doc not in seen_paths: {}, in_seen={}", path, path in seen_paths)
                await self._pipeline.delete_document(info["id"])
                stats["deleted"] += 1
                logger.info("Deleted document: {}", Path(path).name)

        db.commit()
        return stats

    async def index_single_file(
        self,
        file_path: Path,
        min_chunk_size: Optional[int] = None,
        max_chunk_size: Optional[int] = None,
        chunk_overlap_ratio: Optional[float] = None,
    ) -> bool:
        """
        Index a single file (for incremental updates).

        Only processes the given file, doesn't scan the whole directory.
        If the file doesn't exist in the database, adds it; if it exists
        and mtime is newer, updates it.

        Args:
            file_path: File path to index
            min_chunk_size: Optional override for min chunk size
            max_chunk_size: Optional override for max chunk size
            chunk_overlap_ratio: Optional override for overlap ratio

        Returns:
            True if file was indexed (added or updated), False otherwise
        """
        if not file_path.exists() or not file_path.is_file():
            logger.debug("File not found or not a file: {}", file_path)
            return False

        ext = file_path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            logger.debug("Unsupported file type: {}", file_path)
            return False

        db = self._db.db
        abs_path = str(file_path.resolve())
        mtime = file_path.stat().st_mtime

        # Check if file already exists in database
        cursor = db.execute(
            "SELECT id, stored_at FROM documents WHERE path = ?",
            (abs_path,)
        )
        row = cursor.fetchone()

        if row is None:
            # New file, add it
            doc = await self._pipeline.index_document(
                file_path,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size,
                chunk_overlap_ratio=chunk_overlap_ratio,
            )
            db.commit()
            if doc:
                logger.info("Added document: {}", file_path.name)
                return True
            else:
                return False
        else:
            doc_id, stored_at = row
            if mtime > stored_at:
                # File has been updated, update it
                await self._pipeline.delete_document(doc_id)
                doc = await self._pipeline.index_document(
                    file_path,
                    min_chunk_size=min_chunk_size,
                    max_chunk_size=max_chunk_size,
                    chunk_overlap_ratio=chunk_overlap_ratio,
                )
                db.commit()
                if doc:
                    logger.info("Updated document: {}", file_path.name)
                    return True
                else:
                    return False
            else:
                # File unchanged, skip
                logger.debug("File unchanged, skipping: {}", file_path.name)
                return False

    async def schedule_index_update(
        self,
        docs_dir: Path,
        min_chunk_size: Optional[int] = None,
        max_chunk_size: Optional[int] = None,
        chunk_overlap_ratio: Optional[float] = None,
    ) -> None:
        """Schedule an index update after a delay to batch multiple changes."""
        async with self._index_lock:
            if self._index_pending:
                logger.debug("Index update already scheduled")
                return
            self._index_pending = True

        await asyncio.sleep(self._index_delay_seconds)

        async with self._index_lock:
            if not self._index_pending:
                return
            self._index_pending = False

        try:
            await self.scan_and_index(docs_dir, min_chunk_size, max_chunk_size, chunk_overlap_ratio)
            logger.info("Scheduled index update completed")
        except Exception as e:
            logger.error("Scheduled index update failed: {}", e)
            async with self._index_lock:
                self._index_pending = False
