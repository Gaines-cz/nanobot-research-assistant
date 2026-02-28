"""RAG (Retrieval-Augmented Generation) module for nanobot."""

from nanobot.rag.embeddings import EmbeddingProvider, SentenceTransformerEmbeddingProvider
from nanobot.rag.parser import DocumentParser, SemanticChunk
from nanobot.rag.rerank import (
    CrossEncoderReranker,
    RerankService,
    Reranker,
    SemanticDeduplicator,
)
from nanobot.rag.store import (
    ChunkInfo,
    DocumentInfo,
    DocumentStore,
    SearchResult,
    SearchResultWithContext,
)

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
