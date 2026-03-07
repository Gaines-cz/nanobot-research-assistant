"""Unified data models for RAG module.

This module contains all shared data classes used across the RAG system.
All other modules should import from here instead of defining their own.
"""

from dataclasses import dataclass
from typing import List, Optional

# ============================================================================
# Core Domain Models
# ============================================================================

@dataclass
class Document:
    """Document domain model - represents a single document in the store."""
    id: Optional[int] = None
    path: str = ""
    filename: str = ""
    file_type: str = ""
    file_size: Optional[int] = None
    mtime: Optional[float] = None
    stored_at: Optional[float] = None
    title: Optional[str] = None
    doc_type: Optional[str] = None
    abstract: Optional[str] = None


@dataclass
class Chunk:
    """Chunk domain model - represents a single chunk of a document."""
    id: Optional[int] = None
    doc_id: int = 0
    chunk_index: int = 0
    content: str = ""
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None
    chunk_type: Optional[str] = None
    section_title: Optional[str] = None
    granularity: str = "large"
    parent_chunk_id: Optional[int] = None
    embedding: Optional[List[float]] = None


# ============================================================================
# Search Result Models
# ============================================================================

@dataclass
class SearchResult:
    """A single search result (basic version)."""
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
    embedding: Optional[List[float]] = None


@dataclass
class SearchResultWithContext:
    """Search result with expanded context and document info."""
    document: DocumentInfo
    chunk: ChunkInfo
    combined_content: str
    final_score: float
    rank: int


# ============================================================================
# Unified Search Result (New Architecture)
# ============================================================================

@dataclass
class UnifiedSearchResult:
    """
    Unified search result that combines all relevant information.

    This is the primary result type for the new architecture.
    The older types above are kept for backward compatibility.
    """
    document: Document
    chunk: Chunk
    score: float = 0.0
    source: str = "hybrid"  # "vector" | "bm25" | "hybrid"
    rank: int = 0

    # Optional context expansion
    prev_content: Optional[str] = None
    next_content: Optional[str] = None
    combined_content: Optional[str] = None


# ============================================================================
# Compatibility Aliases
# ============================================================================

# Aliases for backward compatibility with evaluation module
BaselineChunkInfo = ChunkInfo
BaselineDocumentInfo = DocumentInfo


@dataclass
class BaselineSearchResult:
    """Minimal search result for baseline evaluation (compatible with judge)."""
    document: DocumentInfo
    chunk: ChunkInfo


# ============================================================================
# Helper Functions
# ============================================================================

def get_result_text(result: SearchResult | SearchResultWithContext) -> str:
    """Get text content from either SearchResult or SearchResultWithContext."""
    if isinstance(result, SearchResultWithContext):
        return result.combined_content
    return result.content


def get_result_score(result: SearchResult | SearchResultWithContext) -> float:
    """Get relevance score from either SearchResult or SearchResultWithContext."""
    if isinstance(result, SearchResultWithContext):
        return result.final_score
    return result.score
