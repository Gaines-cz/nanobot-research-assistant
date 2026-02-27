"""Document store with vector search and full-text search."""

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.rag.embeddings import EmbeddingProvider
from nanobot.rag.parser import DocumentParser


@dataclass
class SearchResult:
    """A single search result."""

    path: str
    filename: str
    chunk_index: int
    content: str
    score: float
    source: str  # "vector" | "fulltext" | "hybrid"


class DocumentStore:
    """
    Unified document store with vector search and full-text search.

    Uses:
    - SQLite for metadata storage
    - sqlite-vec for vector embeddings (optional, falls back to FTS only)
    - SQLite FTS5 for full-text search
    """

    def __init__(self, db_path: Path, embedding_provider: EmbeddingProvider):
        self.db_path = db_path
        self.embedding_provider = embedding_provider
        self._db: sqlite3.Connection | None = None
        self._vector_enabled: bool = False
        self._ensure_db_dir()

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

        # Try to load sqlite-vec extension with graceful fallback
        self._vector_enabled = False
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
        """Initialize the database schema."""
        db = self._get_db()

        # Documents table
        db.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_size INTEGER,
                mtime REAL NOT NULL,
                stored_at REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Chunks table
        db.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                start_pos INTEGER,
                end_pos INTEGER,
                FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE,
                UNIQUE(doc_id, chunk_index)
            )
        """)

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
        now = time.time()

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

        # Insert document
        cursor = db.execute("""
            INSERT INTO documents (path, filename, file_type, file_size, mtime, stored_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (abs_path, path.name, file_type, path.stat().st_size, mtime, now))
        doc_id = cursor.lastrowid

        # Chunk and insert content
        chunks = DocumentParser.chunk_text(content, chunk_size, chunk_overlap)
        if not chunks:
            return

        # Get embeddings only if vector search is enabled
        embeddings = None
        if self._vector_enabled:
            try:
                chunk_contents = [c[0] for c in chunks]
                embeddings = await self.embedding_provider.embed_batch(chunk_contents)
            except Exception as e:
                logger.warning("Could not generate embeddings, vector search disabled: {}", e)
                self._vector_enabled = False
                embeddings = None

        for idx, (chunk_content, start_pos, end_pos) in enumerate(chunks):
            cursor = db.execute("""
                INSERT INTO chunks (doc_id, chunk_index, content, start_pos, end_pos)
                VALUES (?, ?, ?, ?, ?)
            """, (doc_id, idx, chunk_content, start_pos, end_pos))
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
        if self._vector_enabled:
            try:
                # Run both searches in parallel-ish
                vector_results = await self._vector_search(query, top_k * 2)
                fulltext_results = self._fulltext_search(query, top_k * 2)

                # If vector search returned no results, just use fulltext
                if not vector_results:
                    for result in fulltext_results:
                        result.source = "fulltext"
                    return fulltext_results[:top_k]

                # RRF (Reciprocal Rank Fusion)
                # Formula: score = 1 / (k + rank), where k is typically 60
                k = 60
                rrf_scores: dict[str, float] = {}
                sources: dict[str, str] = {}
                result_map: dict[str, SearchResult] = {}

                # Process vector results (priority order preserved)
                for rank, result in enumerate(vector_results, 1):
                    key = f"{result.path}:{result.chunk_index}"
                    rrf_scores[key] = 1.0 / (k + rank)
                    sources[key] = "vector"
                    result_map[key] = result

                # Process fulltext results, accumulate RRF scores
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

                # Build final results with RRF scores
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
                logger.warning("Hybrid search failed, falling back to full-text search only: {}", e)
                self._vector_enabled = False

        # Fallback: full-text search only
        fulltext_results = self._fulltext_search(query, top_k)
        for result in fulltext_results:
            result.source = "fulltext"
        return fulltext_results

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
            # Fallback: simpler query without BM25 (also use sanitized query!)
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
                    score=1.0 - (i * 0.1),  # Simple ranking by order
                    source="fulltext",
                ))

        return results

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the document store."""
        db = self._get_db()

        cursor = db.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]

        cursor = db.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = cursor.fetchone()[0]

        cursor = db.execute("SELECT file_type, COUNT(*) FROM documents GROUP BY file_type")
        by_type = {row[0]: row[1] for row in cursor}

        return {
            "documents": doc_count,
            "chunks": chunk_count,
            "by_file_type": by_type,
        }
