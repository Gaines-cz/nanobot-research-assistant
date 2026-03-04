"""Database connection management for DocumentStore."""

import sqlite3
import time
from pathlib import Path
from typing import Optional

from loguru import logger

from nanobot.config.schema import RAGConfig
from nanobot.rag.embeddings import EmbeddingProvider


class DatabaseConnection:
    """
    Manages SQLite database connection and schema initialization.

    Handles:
    - Connection creation and management
    - sqlite-vec extension loading
    - Schema initialization with backward compatibility
    - FTS5 triggers setup
    """

    def __init__(
        self,
        db_path: Path,
        embedding_provider: Optional[EmbeddingProvider] = None,
        config: Optional[RAGConfig] = None,
    ):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
            embedding_provider: Optional embedding provider for vector search
            config: RAG configuration
        """
        self.db_path = db_path
        self.embedding_provider = embedding_provider
        self.config = config or RAGConfig()

        self._db: Optional[sqlite3.Connection] = None
        self._vector_enabled: bool = False
        self._vector_disabled_at: Optional[float] = None
        self._vector_cooldown_seconds: int = 300

        self._ensure_db_dir()

    def _ensure_db_dir(self) -> None:
        """Ensure the database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def vector_enabled(self) -> bool:
        """Check if vector search is enabled."""
        return self._vector_enabled

    @property
    def db(self) -> sqlite3.Connection:
        """Get or create the database connection."""
        if self._db is not None:
            return self._db
        return self._connect()

    def _connect(self) -> sqlite3.Connection:
        """Create and initialize the database connection."""
        self._db = sqlite3.connect(self.db_path)
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA journal_mode = WAL")

        # Check if we should attempt to re-enable vector search after cooldown
        if not self._vector_enabled and self._vector_disabled_at is not None:
            if time.time() - self._vector_disabled_at >= self._vector_cooldown_seconds:
                logger.info("Attempting to re-enable vector search after cooldown")
                self._vector_disabled_at = None

        # Try to load sqlite-vec extension with graceful fallback
        self._vector_enabled = False
        if self.embedding_provider is not None:
            try:
                self._db.enable_load_extension(True)

                import sqlite_vec

                ext_path = self._get_vec_extension_path(sqlite_vec)
                self._db.load_extension(ext_path)
                self._vector_enabled = True
                logger.debug("sqlite-vec extension loaded, vector search enabled")
            except Exception as e:
                logger.warning("Could not load sqlite-vec extension, vector search disabled: {}", e)
                self._vector_disabled_at = time.time()

        self._init_schema()
        return self._db

    def _get_vec_extension_path(self, sqlite_vec) -> str:
        """Get sqlite-vec extension path, trying different API names."""
        if hasattr(sqlite_vec, 'loadable_path'):
            return sqlite_vec.loadable_path()
        elif hasattr(sqlite_vec, 'extension_path'):
            return sqlite_vec.extension_path()
        elif hasattr(sqlite_vec, 'path'):
            return sqlite_vec.path
        else:
            raise AttributeError("No sqlite-vec path method found")

    def _init_schema(self) -> None:
        """Initialize the database schema with backward compatibility."""
        db = self.db

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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                title TEXT,
                doc_type TEXT,
                abstract TEXT
            )
        """)

        # Add new columns for backward compatibility
        self._add_column_if_not_exists("documents", "title")
        self._add_column_if_not_exists("documents", "doc_type")
        self._add_column_if_not_exists("documents", "abstract")

        # Chunks table
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

        # Add new columns for backward compatibility
        self._add_column_if_not_exists("chunks", "chunk_type")
        self._add_column_if_not_exists("chunks", "section_title")

        # sqlite-vec virtual table for embeddings
        if self._vector_enabled:
            try:
                import sqlite_vec

                dimensions = self.embedding_provider.dimensions
                db = self.db

                # Check if we need to rebuild (by trying to insert test vector)
                need_rebuild = False
                cursor = db.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='chunk_embeddings'
                """)
                if cursor.fetchone():
                    # Table exists, try inserting test vector
                    try:
                        test_embedding = [0.0] * dimensions
                        test_blob = sqlite_vec.serialize_float32(test_embedding)
                        db.execute("""
                            INSERT OR IGNORE INTO chunk_embeddings (chunk_id, embedding)
                            VALUES (?, ?)
                        """, (-999999, test_blob))
                        db.execute("DELETE FROM chunk_embeddings WHERE chunk_id = ?", (-999999,))
                    except (sqlite3.OperationalError, ValueError):
                        # Dimension mismatch, need rebuild
                        need_rebuild = True

                if need_rebuild:
                    logger.warning(
                        "Embedding dimension changed, rebuilding vector table "
                        "(new: {}d)",
                        dimensions
                    )
                    db.execute("DROP TABLE chunk_embeddings")

                # Create table
                db.execute(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings USING vec0(
                        chunk_id INTEGER PRIMARY KEY,
                        embedding FLOAT32[{dimensions}]
                    )
                """)
            except Exception as e:
                logger.warning("Could not create vector table, vector search disabled: {}", e)
                self._vector_enabled = False
                self._vector_disabled_at = time.time()

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
        self._create_fts_triggers(db)

        db.commit()

    def _add_column_if_not_exists(self, table: str, column: str) -> None:
        """Add a column to a table if it doesn't exist."""
        try:
            self.db.execute(f"ALTER TABLE {table} ADD COLUMN {column}")
        except sqlite3.OperationalError:
            pass  # Column already exists

    def _create_fts_triggers(self, db: sqlite3.Connection) -> None:
        """Create FTS5 triggers for automatic index updates."""
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

    def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            try:
                self._db.close()
                logger.debug("Database connection closed")
            except Exception as e:
                logger.warning("Error closing database connection: {}", e)
            finally:
                self._db = None

    def record_vector_disabled(self) -> None:
        """Record when vector search was disabled (for cooldown)."""
        self._vector_disabled_at = time.time()
