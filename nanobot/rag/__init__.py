"""RAG (Retrieval-Augmented Generation) module for nanobot."""

from nanobot.rag.embeddings import EmbeddingProvider, SentenceTransformerEmbeddingProvider
from nanobot.rag.parser import DocumentParser, SemanticChunk
from nanobot.rag.rerank import (
    CrossEncoderReranker,
    Reranker,
    RerankService,
    SemanticDeduplicator,
)
from nanobot.rag.search import (
    ChunkInfo,
    DocumentInfo,
    SearchResult,
    SearchResultWithContext,
)
from nanobot.rag.store import DocumentStore

__all__ = [
    "ChunkInfo",
    "CrossEncoderReranker",
    "DocumentInfo",
    "DocumentParser",
    "DocumentStore",
    "EmbeddingProvider",
    "RerankService",
    "Reranker",
    "SearchResult",
    "SearchResultWithContext",
    "SemanticChunk",
    "SemanticDeduplicator",
    "SentenceTransformerEmbeddingProvider",
]
