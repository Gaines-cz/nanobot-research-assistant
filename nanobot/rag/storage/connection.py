"""Database connection management for DocumentStore.

This is a simplified version that delegates schema initialization
to the storage.schema module.
"""

import sqlite3
import time
from pathlib import Path
from typing import Optional

from loguru import logger

from nanobot.config.schema import RAGConfig
from nanobot.rag.embeddings import EmbeddingProvider
from nanobot.rag.storage.schema import init_schema


class DatabaseConnection:
    """
    Manages SQLite database connection and schema initialization.

    Handles:
    - Connection creation and management
    - sqlite-vec extension loading
    - Delegates schema initialization to storage.schema
    - FTS5 triggers setup (via schema module)
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
            else:
                # Still in cooldown period, skip extension loading attempt
                if self.embedding_provider is not None:
                    logger.debug("Vector search in cooldown, skipping extension load attempt")
                self._init_schema()
                return self._db

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
        """Initialize database schema using the schema module."""
        dimensions = self.embedding_provider.dimensions if self.embedding_provider else None
        self._vector_enabled = init_schema(
            self._db,
            vector_enabled=self._vector_enabled,
            embedding_dimensions=dimensions,
        )

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
        self._vector_enabled = False
        self._vector_disabled_at = time.time()
