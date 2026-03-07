"""Storage layer for RAG - database connections and schema."""

from nanobot.rag.storage.connection import DatabaseConnection
from nanobot.rag.storage.schema import (
    add_column_if_not_exists,
    create_fts_triggers,
    init_schema,
)

__all__ = [
    "DatabaseConnection",
    "init_schema",
    "create_fts_triggers",
    "add_column_if_not_exists",
]
