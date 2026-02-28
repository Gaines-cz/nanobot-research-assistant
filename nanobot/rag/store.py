"""Document store with vector search and full-text search."""

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from nanobot.config.schema import RAGConfig
from nanobot.rag.embeddings import EmbeddingProvider
from nanobot.rag.parser import DocumentParser, SemanticChunk
from nanobot.rag.query import QueryExpander
from nanobot.rag.rerank import CrossEncoderReranker, RerankService


@dataclass
class SearchResult:
    """A single search result."""

    path: str
    filename: str
    chunk_index: int
    content: str
    score: float
    source: str  # "vector" | "fulltext" | "hybrid"


@dataclass
class DocumentInfo:
    """Metadata about a document."""

    id: int
    path: str
    filename: str
    file_type: str
    file_size: Optional[int]
    title: Optional[str] = None
    doc_type: Optional[str] = None
    abstract: Optional[str] = None


@dataclass
class ChunkInfo:
    """Information about a chunk with context."""

    id: int
    doc_id: int
    chunk_index: int
    content: str
    score: float
    source: str
    chunk_type: Optional[str] = None
    section_title: Optional[str] = None
    prev_content: Optional[str] = None
    next_content: Optional[str] = None


@dataclass
class SearchResultWithContext:
    """Search result with expanded context and document info."""

    document: DocumentInfo
    chunk: ChunkInfo
    combined_content: str
    final_score: float
    rank: int


class DocumentStore:
    """
    Unified document store with vector search and full-text search.

    Uses:
    - SQLite for metadata storage
    - sqlite-vec for vector embeddings (optional, falls back to FTS only)
    - SQLite FTS5 for full-text search
    """

    def __init__(self, db_path: Path, embedding_provider: Optional[EmbeddingProvider] = None, config: Optional[RAGConfig] = None):
        self.db_path = db_path
        self.embedding_provider = embedding_provider
        self.config = config or RAGConfig()
        self._db: sqlite3.Connection | None = None
        self._vector_enabled: bool = False
        self._ensure_db_dir()

        # Search cache - stores (timestamp, results)
        self._search_cache: dict[str, tuple[float, list[SearchResultWithContext]]] = {}
        self._basic_search_cache: dict[str, tuple[float, list[SearchResult]]] = {}

        # Initialize rerank service (Phase 4) - only if embedding_provider is available
        self._rerank_service: Optional[RerankService] = None
        if self.config.enable_rerank and embedding_provider is not None:
            self._rerank_service = RerankService(
                reranker=CrossEncoderReranker(self.config.rerank_model),
                rerank_threshold=self.config.rerank_threshold,
                dedup_threshold=self.config.dedup_threshold,
                rerank_top_k=self.config.rerank_top_k,
                enable_rerank=self.config.enable_rerank,
            )

        # Initialize query expander
        self._query_expander = QueryExpander(enabled=self.config.enable_query_expand)

    def _ensure_db_dir(self) -> None:
        """Ensure the database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_db(self) -> sqlite3.Connection:
        """Get or create the database connection."""
        if self._db is not None:
            return self._db

        self._db = sqlite3.connect(self.db_path)
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA journal_mode = WAL")

        # Try to load sqlite-vec extension with graceful fallback (only if embedding_provider is available)
        self._vector_enabled = False
        if self.embedding_provider is not None:
            try:
                # Check if enable_load_extension is available
                if hasattr(self._db, 'enable_load_extension'):
                    self._db.enable_load_extension(True)

                    import sqlite_vec

                    # Try different API names for compatibility
                    if hasattr(sqlite_vec, 'loadable_path'):
                        ext_path = sqlite_vec.loadable_path()
                    elif hasattr(sqlite_vec, 'extension_path'):
                        ext_path = sqlite_vec.extension_path()
                    elif hasattr(sqlite_vec, 'path'):
                        ext_path = sqlite_vec.path
                    else:
                        raise AttributeError("No sqlite-vec path method found")

                    self._db.load_extension(ext_path)
                    self._vector_enabled = True
                    logger.debug("sqlite-vec extension loaded, vector search enabled")
                else:
                    logger.warning("sqlite3 does not support enable_load_extension, vector search disabled")
            except Exception as e:
                logger.warning("Could not load sqlite-vec extension, vector search disabled: {}", e)

        self._init_schema()
        return self._db

    def _init_schema(self) -> None:
        """Initialize the database schema with backward compatibility."""
        db = self._get_db()

        # Documents table - updated with new columns
        db.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_size INTEGER,
                mtime REAL NOT NULL,
                stored_at REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                title TEXT,
                doc_type TEXT,
                abstract TEXT
            )
        """)

        # Add new columns to documents table if they don't exist (backward compatibility)
        try:
            db.execute("ALTER TABLE documents ADD COLUMN title TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        try:
            db.execute("ALTER TABLE documents ADD COLUMN doc_type TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            db.execute("ALTER TABLE documents ADD COLUMN abstract TEXT")
        except sqlite3.OperationalError:
            pass

        # Chunks table - updated with new columns
        db.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                start_pos INTEGER,
                end_pos INTEGER,
                chunk_type TEXT,
                section_title TEXT,
                FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE,
                UNIQUE(doc_id, chunk_index)
            )
        """)

        # Add new columns to chunks table if they don't exist (backward compatibility)
        try:
            db.execute("ALTER TABLE chunks ADD COLUMN chunk_type TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            db.execute("ALTER TABLE chunks ADD COLUMN section_title TEXT")
        except sqlite3.OperationalError:
            pass

        # sqlite-vec virtual table for embeddings (only if vector enabled)
        if self._vector_enabled:
            try:
                # Get embedding dimensions (will trigger model load if needed)
                dimensions = self.embedding_provider.dimensions

                db.execute(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings USING vec0(
                        chunk_id INTEGER PRIMARY KEY,
                        embedding FLOAT32[{dimensions}]
                    )
                """)
            except Exception as e:
                logger.warning("Could not create vector table, vector search disabled: {}", e)
                self._vector_enabled = False

        # FTS5 virtual table for full-text search
        db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content,
                content=chunks,
                content_rowid=id,
                tokenize='porter unicode61'
            )
        """)

        # FTS triggers
        db.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
            END;
        """)
        db.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.id, old.content);
            END;
        """)
        db.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.id, old.content);
                INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
            END;
        """)

        db.commit()

    async def scan_and_index(
        self,
        docs_dir: Path,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> dict[str, int]:
        """
        Scan docs_dir and index documents.

        Returns:
            Dict with counts: {"added": n, "updated": n, "deleted": n}
        """
        if not docs_dir.exists():
            docs_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Created docs directory: {}", docs_dir)
            return {"added": 0, "updated": 0, "deleted": 0}

        db = self._get_db()
        stats = {"added": 0, "updated": 0, "deleted": 0}

        # Get all known documents from DB
        cursor = db.execute("SELECT id, path, mtime, stored_at FROM documents")
        known_docs = {row[1]: {"id": row[0], "mtime": row[2], "stored_at": row[3]} for row in cursor}

        # Scan files in docs_dir
        seen_paths: set[str] = set()

        for file_path in docs_dir.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.name.startswith("."):
                continue

            ext = file_path.suffix.lower()
            if ext not in (".pdf", ".md", ".markdown", ".docx", ".doc", ".txt", ".rst", ".py", ".js", ".ts", ".html", ".css"):
                continue

            abs_path = str(file_path.resolve())
            seen_paths.add(abs_path)
            mtime = file_path.stat().st_mtime

            if abs_path not in known_docs:
                # New document
                await self._add_document(db, file_path, chunk_size, chunk_overlap)
                stats["added"] += 1
                logger.info("Added document: {}", file_path.name)
            elif mtime > known_docs[abs_path]["stored_at"]:
                # Updated document
                await self._update_document(db, known_docs[abs_path]["id"], file_path, chunk_size, chunk_overlap)
                stats["updated"] += 1
                logger.info("Updated document: {}", file_path.name)

        # Delete documents that no longer exist
        for path, info in known_docs.items():
            if path not in seen_paths:
                self._delete_document(db, info["id"])
                stats["deleted"] += 1
                logger.info("Deleted document: {}", Path(path).name)

        db.commit()

        # Clear search cache after index update
        self._search_cache.clear()
        self._basic_search_cache.clear()
        logger.debug("Search cache cleared after index update")

        return stats

    async def _add_document(
        self,
        db: sqlite3.Connection,
        path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        """Add a new document to the store."""
        abs_path = str(path.resolve())
        content, file_type = DocumentParser.parse(path)
        mtime = path.stat().st_mtime
        now = time.time()

        # Extract metadata using new Phase 2 method
        metadata = DocumentParser.extract_metadata(path, content)

        # Insert document with metadata
        cursor = db.execute("""
            INSERT INTO documents (path, filename, file_type, file_size, mtime, stored_at, title, doc_type, abstract)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (abs_path, path.name, file_type, path.stat().st_size, mtime, now,
               metadata.get("title"), metadata.get("doc_type"), metadata.get("abstract")))
        doc_id = cursor.lastrowid

        # Use semantic chunking with phase2b strategy
        min_chunk_size = self.config.min_chunk_size
        max_chunk_size = self.config.max_chunk_size
        overlap_ratio = self.config.chunk_overlap_ratio

        # Check which chunk strategy to use
        if self.config.chunk_strategy == "phase2b":
            semantic_chunks = DocumentParser.chunk_with_section_awareness(
                content, min_chunk_size, max_chunk_size, overlap_ratio
            )
        elif self.config.chunk_strategy == "paragraph":
            semantic_chunks = DocumentParser.chunk_by_paragraph_sentence(
                content, min_chunk_size, max_chunk_size, overlap_ratio
            )
        else:
            # Legacy fixed-size
            chunks = DocumentParser.chunk_text(content, chunk_size, chunk_overlap)
            semantic_chunks = [
                SemanticChunk(content=c[0], start_pos=c[1], end_pos=c[2])
                for c in chunks
            ]

        if not semantic_chunks:
            return

        # Get embeddings only if vector search is enabled
        embeddings = None
        if self._vector_enabled:
            try:
                chunk_contents = [c.content for c in semantic_chunks]
                embeddings = await self.embedding_provider.embed_batch(chunk_contents)
            except Exception as e:
                logger.warning("Could not generate embeddings, vector search disabled: {}", e)
                self._vector_enabled = False
                embeddings = None

        for idx, chunk in enumerate(semantic_chunks):
            cursor = db.execute("""
                INSERT INTO chunks (doc_id, chunk_index, content, start_pos, end_pos, chunk_type, section_title)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (doc_id, idx, chunk.content, chunk.start_pos, chunk.end_pos,
                   chunk.chunk_type, chunk.section_title))
            chunk_id = cursor.lastrowid

            # Insert embedding only if vector search is enabled and we have embeddings
            if self._vector_enabled and embeddings is not None:
                try:
                    import sqlite_vec
                    embedding_blob = sqlite_vec.serialize_float32(embeddings[idx])
                    db.execute("""
                        INSERT OR REPLACE INTO chunk_embeddings (chunk_id, embedding)
                        VALUES (?, ?)
                    """, (chunk_id, embedding_blob))
                except Exception as e:
                    logger.warning("Could not insert embedding, vector search disabled: {}", e)
                    self._vector_enabled = False

    async def _update_document(
        self,
        db: sqlite3.Connection,
        doc_id: int,
        path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        """Update an existing document."""
        self._delete_document(db, doc_id)
        await self._add_document(db, path, chunk_size, chunk_overlap)

    def _delete_document(self, db: sqlite3.Connection, doc_id: int) -> None:
        """Delete a document (cascades to chunks and embeddings)."""
        # Delete from chunk embeddings only if vector search is enabled
        if self._vector_enabled:
            try:
                # Get chunk_ids first to delete from chunk_embeddings
                cursor = db.execute("SELECT id FROM chunks WHERE doc_id = ?", (doc_id,))
                chunk_ids = [row[0] for row in cursor]

                for chunk_id in chunk_ids:
                    db.execute("DELETE FROM chunk_embeddings WHERE chunk_id = ?", (chunk_id,))
            except Exception as e:
                logger.warning("Could not delete from chunk_embeddings: {}", e)

        # Delete from documents (cascades to chunks via FK)
        db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    async def _vector_search(self, query: str, top_k: int) -> list[SearchResult]:
        """Vector similarity search."""
        if not self._vector_enabled:
            return []

        db = self._get_db()
        results: list[SearchResult] = []

        try:
            query_embedding = await self.embedding_provider.embed(query)
            import sqlite_vec
            embedding_blob = sqlite_vec.serialize_float32(query_embedding)

            # Try with distance metric first
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
                # Convert distance to similarity (0-1 range, higher is better)
                # Distance is typically 0-2 for cosine distance
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
            logger.warning("Vector search failed: {}", e)
            self._vector_enabled = False
            return []

        return results

    def _sanitize_fts_query(self, query: str) -> str:
        """Sanitize query for FTS5, escaping special characters."""
        # FTS5 special characters: " ( ) * # ^ - : {}
        # Remove them to avoid syntax errors
        special_chars = ['"', "(", ")", "*", "#", "^", "-", ":", "{", "}"]
        sanitized = query
        for char in special_chars:
            sanitized = sanitized.replace(char, " ")
        # Collapse multiple spaces
        import re
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        return sanitized

    def _fulltext_search(self, query: str, top_k: int) -> list[SearchResult]:
        """Full-text search using FTS5."""
        db = self._get_db()
        results: list[SearchResult] = []

        # Sanitize query to avoid FTS5 syntax errors
        safe_query = self._sanitize_fts_query(query)

        # If sanitized query is empty, fall back to getting recent chunks
        if not safe_query:
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
            # First try with BM25 scoring
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

            for row in cursor:
                bm25_score = row[4] if row[4] is not None else 1.0
                # Normalize BM25 score (lower BM25 is better, so invert)
                normalized = 1.0 / (1.0 + bm25_score)
                results.append(SearchResult(
                    path=row[0],
                    filename=row[1],
                    chunk_index=row[2],
                    content=row[3],
                    score=normalized,
                    source="fulltext",
                ))
        except Exception as e:
            logger.warning("FTS with BM25 failed, trying simpler query: {}", e)
            try:
                # Fallback 1: simpler query without BM25
                cursor = db.execute("""
                    SELECT
                        d.path,
                        d.filename,
                        c.chunk_index,
                        c.content
                    FROM chunks_fts
                    JOIN chunks c ON chunks_fts.rowid = c.id
                    JOIN documents d ON c.doc_id = d.id
                    WHERE chunks_fts MATCH ?
                    LIMIT ?
                """, (safe_query, top_k))
                for i, row in enumerate(cursor):
                    results.append(SearchResult(
                        path=row[0],
                        filename=row[1],
                        chunk_index=row[2],
                        content=row[3],
                        score=1.0 - (i * 0.1),
                        source="fulltext",
                    ))
            except Exception as e2:
                logger.warning("FTS fallback also failed, using non-FTS query: {}", e2)
                # Fallback 2: use LIKE query instead of FTS
                like_query = f"%{safe_query}%"
                cursor = db.execute("""
                    SELECT
                        d.path,
                        d.filename,
                        c.chunk_index,
                        c.content
                    FROM chunks c
                    JOIN documents d ON c.doc_id = d.id
                    WHERE c.content LIKE ?
                    LIMIT ?
                """, (like_query, top_k))
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

    def close(self) -> None:
        """Close the database connection if open."""
        if self._db is not None:
            try:
                self._db.close()
                logger.debug("Database connection closed")
            except Exception as e:
                logger.warning("Error closing database connection: {}", e)
            finally:
                self._db = None

    def is_vector_enabled(self) -> bool:
        """Check if vector search is enabled."""
        return self._vector_enabled

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the document store."""
        db = self._get_db()

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

        # Last scan time (most recent stored_at)
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
            "vector_enabled": self._vector_enabled,
        }

    async def _step1_core_chunk_recall(self, query: str) -> list[SearchResult]:
        """
        Step 1: Core chunk recall using BM25 + vector search.
        Applies thresholds and returns Top 5 results.
        """
        top_k = 5
        bm25_threshold = self.config.bm25_threshold
        vector_threshold = self.config.vector_threshold

        if self._vector_enabled:
            try:
                vector_results = await self._vector_search(query, top_k * 2)
                fulltext_results = self._fulltext_search(query, top_k * 2)

                # Filter by thresholds
                filtered_vector = [r for r in vector_results if r.score >= vector_threshold]
                filtered_ft = [r for r in fulltext_results if r.score >= bm25_threshold]

                # RRF fusion
                k = self.config.rrf_k
                rrf_scores: dict[str, float] = {}
                sources: dict[str, str] = {}
                result_map: dict[str, SearchResult] = {}

                for rank, result in enumerate(filtered_vector, 1):
                    key = f"{result.path}:{result.chunk_index}"
                    rrf_scores[key] = 1.0 / (k + rank)
                    sources[key] = "vector"
                    result_map[key] = result

                for rank, result in enumerate(filtered_ft, 1):
                    key = f"{result.path}:{result.chunk_index}"
                    score = 1.0 / (k + rank)
                    if key in rrf_scores:
                        rrf_scores[key] += score
                        sources[key] = "hybrid"
                    else:
                        rrf_scores[key] = score
                        sources[key] = "fulltext"
                        result_map[key] = result

                results: list[SearchResult] = []
                for key, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
                    base_result = result_map[key]
                    results.append(SearchResult(
                        path=base_result.path,
                        filename=base_result.filename,
                        chunk_index=base_result.chunk_index,
                        content=base_result.content,
                        score=rrf_score,
                        source=sources[key],
                    ))

                return results[:top_k]
            except Exception as e:
                logger.warning("Hybrid search failed, falling back: {}", e)

        # Fallback: full-text only
        fulltext_results = self._fulltext_search(query, top_k)
        filtered = [r for r in fulltext_results if r.score >= bm25_threshold]
        return filtered[:top_k] if filtered else fulltext_results[:top_k]

    def _step2_context_expansion(self, core_results: list[SearchResult]) -> list[ChunkInfo]:
        """
        Step 2: Expand context around core chunks.
        Returns chunks with previous and next chunks included.
        """
        if not self.config.enable_context_expansion:
            # Return without expansion
            chunks = []
            db = self._get_db()
            for result in core_results:
                cursor = db.execute("""
                    SELECT c.id, c.doc_id, c.chunk_index, c.content, c.chunk_type, c.section_title
                    FROM chunks c
                    JOIN documents d ON c.doc_id = d.id
                    WHERE d.path = ? AND c.chunk_index = ?
                """, (result.path, result.chunk_index))
                row = cursor.fetchone()
                if row:
                    chunks.append(ChunkInfo(
                        id=row[0],
                        doc_id=row[1],
                        chunk_index=row[2],
                        content=row[3],
                        score=result.score,
                        source=result.source,
                        chunk_type=row[4],
                        section_title=row[5],
                    ))
            return chunks

        db = self._get_db()
        expanded_chunks = []
        prev_count = self.config.context_prev_chunks
        next_count = self.config.context_next_chunks

        for result in core_results:
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

            # Get prev_count chunks before
            prev_contents = []
            for i in range(1, prev_count + 1):
                cursor = db.execute("""
                    SELECT content FROM chunks
                    WHERE doc_id = ? AND chunk_index = ?
                """, (doc_id, chunk_idx - i))
                prev_row = cursor.fetchone()
                if prev_row:
                    prev_contents.insert(0, prev_row[0])  # Insert at beginning to maintain order
            prev_content = "\n\n".join(prev_contents) if prev_contents else None

            # Get next_count chunks after
            next_contents = []
            for i in range(1, next_count + 1):
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
            ))

        return expanded_chunks

    def _step3_document_level(self, core_results: list[SearchResult]) -> list[tuple[DocumentInfo, float]]:
        """
        Step 3: Document-level retrieval.
        Returns Top 3 documents by average score of their chunks.
        """
        if not self.config.enable_document_level:
            return []

        db = self._get_db()
        doc_scores: dict[int, list[float]] = {}
        doc_info_map: dict[int, DocumentInfo] = {}

        for result in core_results:
            cursor = db.execute("""
                SELECT id, path, filename, file_type, file_size, title, doc_type, abstract
                FROM documents WHERE path = ?
            """, (result.path,))
            row = cursor.fetchone()
            if row:
                doc_id = row[0]
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = []
                    doc_info_map[doc_id] = DocumentInfo(
                        id=row[0],
                        path=row[1],
                        filename=row[2],
                        file_type=row[3],
                        file_size=row[4],
                        title=row[5],
                        doc_type=row[6],
                        abstract=row[7],
                    )
                doc_scores[doc_id].append(result.score)

        doc_avg_scores = []
        for doc_id, scores in doc_scores.items():
            avg_score = sum(scores) / len(scores)
            doc_avg_scores.append((doc_info_map[doc_id], avg_score))

        doc_avg_scores.sort(key=lambda x: x[1], reverse=True)
        return doc_avg_scores[:self.config.top_documents]

    def _merge_context_and_document_results(
        self,
        expanded_chunks: list[ChunkInfo],
        top_docs: list[tuple[DocumentInfo, float]]
    ) -> list[SearchResultWithContext]:
        """Merge context-expanded chunks with document-level results."""
        db = self._get_db()
        results = []

        doc_info_cache: dict[int, DocumentInfo] = {}
        for doc, _ in top_docs:
            doc_info_cache[doc.id] = doc

        for i, chunk in enumerate(expanded_chunks):
            if chunk.doc_id not in doc_info_cache:
                cursor = db.execute("""
                    SELECT id, path, filename, file_type, file_size, title, doc_type, abstract
                    FROM documents WHERE id = ?
                """, (chunk.doc_id,))
                row = cursor.fetchone()
                if row:
                    doc_info_cache[chunk.doc_id] = DocumentInfo(
                        id=row[0],
                        path=row[1],
                        filename=row[2],
                        file_type=row[3],
                        file_size=row[4],
                        title=row[5],
                        doc_type=row[6],
                        abstract=row[7],
                    )

            doc_info = doc_info_cache.get(chunk.doc_id)
            if not doc_info:
                continue

            combined_parts = []
            if chunk.prev_content:
                combined_parts.append(chunk.prev_content)
            combined_parts.append(chunk.content)
            if chunk.next_content:
                combined_parts.append(chunk.next_content)
            combined_content = "\n\n".join(combined_parts)

            doc_bonus = 0.0
            for doc, score in top_docs:
                if doc.id == chunk.doc_id:
                    doc_bonus = score * 0.1
                    break

            final_score = chunk.score + doc_bonus

            results.append(SearchResultWithContext(
                document=doc_info,
                chunk=chunk,
                combined_content=combined_content,
                final_score=final_score,
                rank=i + 1,
            ))

        results.sort(key=lambda x: x.final_score, reverse=True)
        for i, r in enumerate(results):
            r.rank = i + 1

        return results

    async def _apply_rerank(
        self,
        query: str,
        results: list[SearchResultWithContext],
    ) -> list[SearchResultWithContext]:
        """
        Phase 4: Apply Cross-Encoder reranking.

        Only reranks top-k results (config.rerank_top_k) for M4 optimization.
        Keeps only results with score >= rerank_threshold.
        """
        if not self._rerank_service or not results:
            return results

        try:
            # Prepare candidates for rerank: (text, metadata, embedding)
            candidates: list[tuple[str, Any, list[float]]] = []
            result_map: dict[int, SearchResultWithContext] = {}

            for i, result in enumerate(results):
                candidates.append((
                    result.combined_content,
                    result,
                    []  # We'll need to get embeddings
                ))
                result_map[i] = result

            # Get embeddings for all candidates (if needed for dedup)
            if self._vector_enabled:
                try:
                    texts = [c[0] for c in candidates]
                    embeddings = await self.embedding_provider.embed_batch(texts)
                    candidates = [
                        (c[0], c[1], emb)
                        for c, emb in zip(candidates, embeddings)
                    ]
                except Exception as e:
                    logger.warning("Could not get embeddings for rerank/dedup: {}", e)

            # Apply rerank and dedup
            reranked = await self._rerank_service.rerank_and_dedup(query, candidates)

            # Map back to SearchResultWithContext with updated scores
            final_results: list[SearchResultWithContext] = []
            for original_idx, result, new_score in reranked:
                # Update the final score
                result.final_score = new_score
                final_results.append(result)

            # Re-rank
            for i, r in enumerate(final_results):
                r.rank = i + 1

            return final_results

        except Exception as e:
            logger.warning("Rerank failed, returning original results: {}", e)
            return results

    async def search_advanced(self, query: str) -> list[SearchResultWithContext]:
        """
        Advanced multi-step search pipeline (Phase 1-4):
        1. Core chunk recall (BM25 + vector, dual thresholds, Top5)
        2. Context expansion (prev1 + core + next1)
        3. Document-level prioritization (Top3 docs)
        4. Cross-Encoder rerank (Top20, M4 optimized) + semantic dedup
        """
        # Check cache first
        cache_key = f"{hash(query)}"
        if self.config.enable_search_cache:
            if cache_key in self._search_cache:
                ts, cached_results = self._search_cache[cache_key]
                if time.time() - ts < self.config.cache_ttl_seconds:
                    logger.debug("Advanced search cache hit for query: {}", query)
                    return cached_results

        # Expand query (abbreviations, synonyms)
        expanded_query = self._query_expander.expand(query)

        # Step 1-3: Core recall -> Context expansion -> Document-level -> Merge
        core_results = await self._step1_core_chunk_recall(expanded_query)
        if not core_results:
            # Cache empty results too
            if self.config.enable_search_cache:
                self._search_cache[cache_key] = (time.time(), [])
            return []

        expanded_chunks = self._step2_context_expansion(core_results)
        top_docs = self._step3_document_level(core_results)
        merged_results = self._merge_context_and_document_results(expanded_chunks, top_docs)

        # Step 4: Apply rerank and dedup (Phase 4)
        if self.config.enable_rerank and self._rerank_service:
            final_results = await self._apply_rerank(expanded_query, merged_results)
        else:
            final_results = merged_results

        # Store in cache
        if self.config.enable_search_cache:
            self._search_cache[cache_key] = (time.time(), final_results)

        return final_results

    async def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Hybrid search: vector + full-text with RRF (Reciprocal Rank Fusion) reranking.
        Falls back to full-text search only if vector search is not available.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of SearchResult sorted by relevance
        """
        # Check cache first
        cache_key = f"{hash(query)}:{top_k}"
        if self.config.enable_search_cache:
            if cache_key in self._basic_search_cache:
                ts, cached_results = self._basic_search_cache[cache_key]
                if time.time() - ts < self.config.cache_ttl_seconds:
                    logger.debug("Basic search cache hit for query: {}", query)
                    return cached_results

        # Expand query (abbreviations, synonyms)
        expanded_query = self._query_expander.expand(query)
        results: list[SearchResult] = []

        if self._vector_enabled:
            try:
                vector_results = await self._vector_search(expanded_query, top_k * 2)
                fulltext_results = self._fulltext_search(expanded_query, top_k * 2)

                if not vector_results:
                    for result in fulltext_results:
                        result.source = "fulltext"
                    results = fulltext_results[:top_k]
                else:
                    k = self.config.rrf_k
                    rrf_scores: dict[str, float] = {}
                    sources: dict[str, str] = {}
                    result_map: dict[str, SearchResult] = {}

                    for rank, result in enumerate(vector_results, 1):
                        key = f"{result.path}:{result.chunk_index}"
                        rrf_scores[key] = 1.0 / (k + rank)
                        sources[key] = "vector"
                        result_map[key] = result

                    for rank, result in enumerate(fulltext_results, 1):
                        key = f"{result.path}:{result.chunk_index}"
                        score = 1.0 / (k + rank)
                        if key in rrf_scores:
                            rrf_scores[key] += score
                            sources[key] = "hybrid"
                        else:
                            rrf_scores[key] = score
                            sources[key] = "fulltext"
                            result_map[key] = result

                    for key, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
                        base_result = result_map[key]
                        results.append(SearchResult(
                            path=base_result.path,
                            filename=base_result.filename,
                            chunk_index=base_result.chunk_index,
                            content=base_result.content,
                            score=rrf_score,
                            source=sources[key],
                        ))

                    results = results[:top_k]
            except Exception as e:
                logger.warning("Hybrid search failed, falling back to full-text search only: {}", e)
                self._vector_enabled = False
                # Fall through to full-text only

        if not results:
            fulltext_results = self._fulltext_search(expanded_query, top_k)
            for result in fulltext_results:
                result.source = "fulltext"
            results = fulltext_results

        # Store in cache
        if self.config.enable_search_cache:
            self._basic_search_cache[cache_key] = (time.time(), results)

        return results
