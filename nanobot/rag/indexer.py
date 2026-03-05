"""Document indexing for DocumentStore."""

import asyncio
import sqlite3
import time
from pathlib import Path
from typing import Optional

from loguru import logger

from nanobot.config.schema import RAGConfig
from nanobot.rag.connection import DatabaseConnection
from nanobot.rag.embeddings import EmbeddingProvider
from nanobot.rag.parser import DocumentParser, SemanticChunk


class DocumentIndexer:
    """
    Handles document indexing operations.

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

        # Batch index update
        self._index_pending: bool = False
        self._index_lock = asyncio.Lock()
        self._index_delay_seconds: float = 30.0

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
                # New document - validate before adding
                if not self._validate_file(file_path):
                    logger.warning("Skipping invalid file: {}", file_path.name)
                    stats["skipped"] = stats.get("skipped", 0) + 1
                    continue
                success = await self._add_document(db, file_path, chunk_size, chunk_overlap_ratio)
                if success:
                    stats["added"] += 1
                    logger.info("Added document: {}", file_path.name)
                else:
                    stats["skipped"] += 1
            elif mtime > known_docs[abs_path]["stored_at"]:
                # Updated document - validate before updating (skip update if invalid to preserve old data)
                if not self._validate_file(file_path):
                    logger.warning("Skipping invalid file, preserving old version: {}", file_path.name)
                    stats["skipped"] = stats.get("skipped", 0) + 1
                    continue
                success = await self._update_document(db, known_docs[abs_path]["id"], file_path, chunk_size, chunk_overlap_ratio)
                if success:
                    stats["updated"] += 1
                    logger.info("Updated document: {}", file_path.name)
                else:
                    stats["skipped"] += 1

        # Delete documents that no longer exist
        logger.debug("Seen paths during scan: {}", seen_paths)
        for path, info in known_docs.items():
            if path not in seen_paths:
                logger.debug("Deleting doc not in seen_paths: {}, in_seen={}", path, path in seen_paths)
                self._delete_document(db, info["id"])
                stats["deleted"] += 1
                logger.info("Deleted document: {}", Path(path).name)

        db.commit()
        return stats

    def _validate_file(self, path: Path) -> bool:
        """
        Validate file format before parsing.

        Checks:
        - PDF files must start with %PDF (possibly after UTF-8 BOM or leading whitespace)
        - HTML files should not be saved as PDF
        - Other text files should have valid encoding

        Args:
            path: Path to the file to validate

        Returns:
            True if file appears valid, False otherwise
        """
        ext = path.suffix.lower()

        if ext == ".pdf":
            try:
                with open(path, "rb") as f:
                    # Read first 10 bytes to cover BOM + header
                    header = f.read(10)
                    # Skip UTF-8 BOM if present
                    if header.startswith(b"\xef\xbb\xbf"):
                        header = header[3:]
                    # Skip any leading whitespace
                    header = header.lstrip()
                    # Check for PDF magic bytes
                    if not header.startswith(b"%PDF"):
                        # Check if it's HTML (common download error) - support case-insensitive
                        header_lower = header.lower()
                        if header_lower.startswith(b"<!do") or header_lower.startswith(b"<htm"):
                            logger.warning("File appears to be HTML, not PDF: {}", path.name)
                            return False
                        logger.warning("Invalid PDF header: {}", path.name)
                        return False
            except Exception as e:
                logger.warning("Error validating file {}: {}", path.name, e)
                return False

        return True

    async def index_single_file(
        self,
        file_path: Path,
        chunk_size: Optional[int] = None,
        chunk_overlap_ratio: Optional[float] = None,
    ) -> bool:
        """
        索引单个文件（增量更新用）。

        只处理给定的文件，不扫描整个目录。
        如果文件不存在于数据库，添加它；如果已存在且 mtime 更新，更新它。

        Args:
            file_path: 要索引的文件路径
            chunk_size: 可选的分块大小覆盖
            chunk_overlap_ratio: 可选的分块重叠率覆盖

        Returns:
            True: 文件被成功索引（新增或更新）
            False: 文件未变化、无效、或处理失败
        """
        if not file_path.exists() or not file_path.is_file():
            logger.debug("File not found or not a file: {}", file_path)
            return False

        ext = file_path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            logger.debug("Unsupported file type: {}", file_path)
            return False

        # Validate file before processing
        if not self._validate_file(file_path):
            logger.warning("Skipping invalid file: {}", file_path.name)
            return False

        db = self._db.db
        abs_path = str(file_path.resolve())
        mtime = file_path.stat().st_mtime

        # 检查数据库中是否已有此文件
        cursor = db.execute(
            "SELECT id, stored_at FROM documents WHERE path = ?",
            (abs_path,)
        )
        row = cursor.fetchone()

        if row is None:
            # 新文件，添加
            success = await self._add_document(db, file_path, chunk_size, chunk_overlap_ratio)
            db.commit()
            if success:
                logger.info("Added document: {}", file_path.name)
                return True
            else:
                return False
        else:
            doc_id, stored_at = row
            if mtime > stored_at:
                # 文件已更新，更新
                success = await self._update_document(db, doc_id, file_path, chunk_size, chunk_overlap_ratio)
                db.commit()
                if success:
                    logger.info("Updated document: {}", file_path.name)
                    return True
                else:
                    return False
            else:
                # 文件未变化，跳过
                logger.debug("File unchanged, skipping: {}", file_path.name)
                return False

    async def schedule_index_update(
        self,
        docs_dir: Path,
        chunk_size: Optional[int] = None,
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
            await self.scan_and_index(docs_dir, chunk_size, chunk_overlap_ratio)
            logger.info("Scheduled index update completed")
        except Exception as e:
            logger.error("Scheduled index update failed: {}", e)
            async with self._index_lock:
                self._index_pending = False

    async def _add_document(
        self,
        db: sqlite3.Connection,
        path: Path,
        chunk_size: Optional[int] = None,
        chunk_overlap_ratio: Optional[float] = None,
    ) -> bool:
        """
        Add a new document to the store.

        Returns:
            True if document was successfully added, False otherwise.
        """

        abs_path = str(path.resolve())

        try:
            content, file_type = DocumentParser.parse(path)
        except Exception as e:
            logger.warning("Failed to parse file {}, skipping: {}", path.name, e)
            return False

        mtime = path.stat().st_mtime
        now = time.time()

        # Extract metadata
        metadata = DocumentParser.extract_metadata(path, content)

        # Insert document with metadata
        cursor = db.execute("""
            INSERT INTO documents (path, filename, file_type, file_size, mtime, stored_at, title, doc_type, abstract)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (abs_path, path.name, file_type, path.stat().st_size, mtime, now,
               metadata.get("title"), metadata.get("doc_type"), metadata.get("abstract")))
        doc_id = cursor.lastrowid

        # Use provided parameters or fall back to config
        max_chunk_size = chunk_size if chunk_size is not None else self.config.max_chunk_size
        min_chunk_size = max_chunk_size // 2
        overlap_ratio = chunk_overlap_ratio if chunk_overlap_ratio is not None else self.config.chunk_overlap_ratio

        # Chunk the document
        semantic_chunks = self._chunk_content(content, min_chunk_size, max_chunk_size, overlap_ratio)

        if not semantic_chunks:
            # No chunks generated - rollback the document record to avoid orphaned entries
            db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            logger.warning("No chunks generated for file {}, skipping: {}", path.name, path)
            return False

        # Get embeddings
        embeddings = None
        if self._db.vector_enabled:
            try:
                chunk_contents = [c.content for c in semantic_chunks]
                embeddings = await self._embedding_provider.embed_batch(chunk_contents)
            except Exception as e:
                logger.warning("Could not generate embeddings, vector search disabled: {}", e)
                self._db.record_vector_disabled()
                embeddings = None

        # Insert chunks
        for idx, chunk in enumerate(semantic_chunks):
            cursor = db.execute("""
                INSERT INTO chunks (doc_id, chunk_index, content, start_pos, end_pos, chunk_type, section_title)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (doc_id, idx, chunk.content, chunk.start_pos, chunk.end_pos,
                   chunk.chunk_type, chunk.section_title))
            chunk_id = cursor.lastrowid

            # Insert embedding
            if self._db.vector_enabled and embeddings is not None:
                try:
                    import sqlite_vec
                    embedding_blob = sqlite_vec.serialize_float32(embeddings[idx])
                    db.execute("""
                        INSERT OR REPLACE INTO chunk_embeddings (chunk_id, embedding)
                        VALUES (?, ?)
                    """, (chunk_id, embedding_blob))
                except Exception as e:
                    logger.warning("Could not insert embedding, vector search disabled: {}", e)
                    self._db.record_vector_disabled()

        return True

    async def _update_document(
        self,
        db: sqlite3.Connection,
        doc_id: int,
        path: Path,
        chunk_size: Optional[int] = None,
        chunk_overlap_ratio: Optional[float] = None,
    ) -> bool:
        """
        Update an existing document.

        Returns:
            True if document was successfully updated, False otherwise.
            On failure, the old document is preserved (not deleted).
        """
        # First, verify the new file can be parsed (pre-validation)
        # This avoids deleting the old document only to find the new one is invalid
        try:
            content, file_type = DocumentParser.parse(path)
        except Exception as e:
            logger.warning("Failed to parse file {}, skipping: {}", path.name, e)
            return False

        # Extract metadata
        metadata = DocumentParser.extract_metadata(path, content)

        # Use provided parameters or fall back to config
        max_chunk_size = chunk_size if chunk_size is not None else self.config.max_chunk_size
        min_chunk_size = max_chunk_size // 2
        overlap_ratio = chunk_overlap_ratio if chunk_overlap_ratio is not None else self.config.chunk_overlap_ratio

        # Chunk the document
        semantic_chunks = self._chunk_content(content, min_chunk_size, max_chunk_size, overlap_ratio)

        if not semantic_chunks:
            logger.warning("No chunks generated for file {}, skipping: {}", path.name, path)
            return False

        # Pre-validation passed - safe to update
        abs_path = str(path.resolve())
        mtime = path.stat().st_mtime
        now = time.time()

        # Update document metadata
        try:
            db.execute("""
                UPDATE documents
                SET path = ?, filename = ?, file_type = ?, file_size = ?, mtime = ?, stored_at = ?, title = ?, doc_type = ?, abstract = ?
                WHERE id = ?
            """, (abs_path, path.name, file_type, path.stat().st_size, mtime, now,
                  metadata.get("title"), metadata.get("doc_type"), metadata.get("abstract"), doc_id))
        except Exception as e:
            logger.warning("Failed to update document {}, error: {}", path.name, e)
            return False

        # Delete old chunks (embeddings will be deleted via CASCADE)
        db.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

        # Get embeddings
        embeddings = None
        if self._db.vector_enabled:
            try:
                chunk_contents = [c.content for c in semantic_chunks]
                embeddings = await self._embedding_provider.embed_batch(chunk_contents)
            except Exception as e:
                logger.warning("Could not generate embeddings, vector search disabled: {}", e)
                self._db.record_vector_disabled()
                embeddings = None

        # Insert new chunks
        try:
            for idx, chunk in enumerate(semantic_chunks):
                cursor = db.execute("""
                    INSERT INTO chunks (doc_id, chunk_index, content, start_pos, end_pos, chunk_type, section_title)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (doc_id, idx, chunk.content, chunk.start_pos, chunk.end_pos,
                       chunk.chunk_type, chunk.section_title))
                chunk_id = cursor.lastrowid

                # Insert embedding
                if self._db.vector_enabled and embeddings is not None:
                    try:
                        import sqlite_vec
                        embedding_blob = sqlite_vec.serialize_float32(embeddings[idx])
                        db.execute("""
                            INSERT OR REPLACE INTO chunk_embeddings (chunk_id, embedding)
                            VALUES (?, ?)
                        """, (chunk_id, embedding_blob))
                    except Exception as e:
                        logger.warning("Could not insert embedding, vector search disabled: {}", e)
                        self._db.record_vector_disabled()
        except Exception as e:
            logger.warning("Failed to insert chunks for {}, error: {}", path.name, e)
            return False

        return True

    def _delete_document(self, db: sqlite3.Connection, doc_id: int) -> None:
        """Delete a document (cascades to chunks and embeddings)."""
        # Delete from chunk embeddings
        if self._db.vector_enabled:
            try:
                cursor = db.execute("SELECT id FROM chunks WHERE doc_id = ?", (doc_id,))
                chunk_ids = [row[0] for row in cursor]
                for chunk_id in chunk_ids:
                    db.execute("DELETE FROM chunk_embeddings WHERE chunk_id = ?", (chunk_id,))
            except Exception as e:
                logger.warning("Could not delete from chunk_embeddings: {}", e)

        # Delete from documents (cascades to chunks via FK)
        db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    def _chunk_content(
        self,
        content: str,
        min_chunk_size: int,
        max_chunk_size: int,
        overlap_ratio: float,
    ) -> list[SemanticChunk]:
        """Chunk content using configured strategy."""
        if self.config.chunk_strategy == "phase2b":
            return DocumentParser.chunk_with_section_awareness(
                content, min_chunk_size, max_chunk_size, overlap_ratio
            )
        elif self.config.chunk_strategy == "paragraph":
            return DocumentParser.chunk_by_paragraph_sentence(
                content, min_chunk_size, max_chunk_size, overlap_ratio
            )
        else:
            # Legacy fixed-size
            legacy_overlap = int(max_chunk_size * overlap_ratio)
            chunks = DocumentParser.chunk_text(content, max_chunk_size, legacy_overlap)
            return [
                SemanticChunk(content=c[0], start_pos=c[1], end_pos=c[2])
                for c in chunks
            ]
