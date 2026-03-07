"""Database schema definitions for RAG.

This module contains all database schema-related functionality,
extracted from connection.py for better separation of concerns.
"""

import sqlite3
from typing import Optional

from loguru import logger


def init_schema(
    db: sqlite3.Connection,
    vector_enabled: bool = False,
    embedding_dimensions: Optional[int] = None,
    get_vec_extension_path: Optional[callable] = None,
) -> bool:
    """
    Initialize the database schema.

    Args:
        db: SQLite database connection
        vector_enabled: Whether vector search is enabled
        embedding_dimensions: Dimension size for embeddings (required if vector_enabled)
        get_vec_extension_path: Optional function to get sqlite-vec extension path

    Returns:
        True if vector remains enabled, False if disabled due to errors
    """
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
    add_column_if_not_exists(db, "documents", "title")
    add_column_if_not_exists(db, "documents", "doc_type")
    add_column_if_not_exists(db, "documents", "abstract")

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
            granularity TEXT NOT NULL DEFAULT 'large',
            parent_chunk_id INTEGER,
            FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE,
            FOREIGN KEY (parent_chunk_id) REFERENCES chunks(id) ON DELETE CASCADE,
            UNIQUE(doc_id, chunk_index, granularity)
        )
    """)

    # Add new columns for backward compatibility
    add_column_if_not_exists(db, "chunks", "chunk_type")
    add_column_if_not_exists(db, "chunks", "section_title")
    add_column_if_not_exists(db, "chunks", "granularity TEXT NOT NULL DEFAULT 'large'")
    add_column_if_not_exists(db, "chunks", "parent_chunk_id INTEGER")

    # Add indexes for dual granularity
    db.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_granularity ON chunks(doc_id, granularity)
    """)
    db.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_chunk_id)
    """)

    # sqlite-vec virtual table for embeddings
    if vector_enabled and embedding_dimensions:
        try:
            import sqlite_vec

            # Check if we need to rebuild (by trying to insert test vector)
            need_rebuild = False
            cursor = db.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='chunk_embeddings'
            """)
            if cursor.fetchone():
                # Table exists, try inserting test vector
                try:
                    test_embedding = [0.0] * embedding_dimensions
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
                    embedding_dimensions
                )
                db.execute("DROP TABLE chunk_embeddings")

            # Create table
            db.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings USING vec0(
                    chunk_id INTEGER PRIMARY KEY,
                    embedding FLOAT32[{embedding_dimensions}]
                )
            """)
        except Exception as e:
            logger.warning("Could not create vector table, vector search disabled: {}", e)
            vector_enabled = False

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
    create_fts_triggers(db)

    db.commit()
    return vector_enabled


def add_column_if_not_exists(db: sqlite3.Connection, table: str, column: str) -> None:
    """Add a column to a table if it doesn't exist."""
    try:
        db.execute(f"ALTER TABLE {table} ADD COLUMN {column}")
    except sqlite3.OperationalError:
        pass  # Column already exists


def create_fts_triggers(db: sqlite3.Connection) -> None:
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
