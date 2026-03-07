"""RAG (Retrieval-Augmented Generation) module for nanobot.

Refactored architecture:
- models.py - Unified data models
- core/interfaces.py - Core abstract interfaces
- storage/ - Database connection and schema
- indexing/ - Document parsing and chunking
- retrieval/ - Search, rerank, and context expansion
- evaluation/ - RAG evaluation tools
"""

# ============================================================================
# Data Models
# ============================================================================

# ============================================================================
# Other Components
# ============================================================================
from nanobot.rag.cache import SearchCacheManager

# ============================================================================
# Core Interfaces
# ============================================================================
from nanobot.rag.core.interfaces import (
    Chunker,
    DocumentParser,
    Index,
    Reranker,
)
from nanobot.rag.embeddings import (
    EmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)

# ============================================================================
# Evaluation
# ============================================================================
from nanobot.rag.evaluation import (
    DataGenerator,
    EvalConfig,
    EvalQuery,
    EvalResult,
    EvalSummary,
    MetricsCalculator,
    RAGEvaluator,
    ResultJudge,
)

# ============================================================================
# Indexing Layer
# ============================================================================
from nanobot.rag.indexing.chunker import (
    HierarchicalChunker,
    HierarchicalChunks,
    SemanticChunk,
    SemanticChunker,
)
from nanobot.rag.indexing.indexer import DocumentIndexer
from nanobot.rag.indexing.parser import DocumentParser as IndexingDocumentParser
from nanobot.rag.indexing.pipeline import IndexingPipeline
from nanobot.rag.models import (
    BaselineChunkInfo,
    BaselineDocumentInfo,
    BaselineSearchResult,
    Chunk,
    ChunkInfo,
    Document,
    DocumentInfo,
    SearchResult,
    SearchResultWithContext,
    UnifiedSearchResult,
)
from nanobot.rag.query import QueryExpander
from nanobot.rag.retrieval.base import (
    AdvancedRetriever,
    Retriever,
)

# ============================================================================
# Retrieval Layer
# ============================================================================
from nanobot.rag.retrieval.base import (
    AdvancedRetriever as AdvancedRetrieverBase,
)
from nanobot.rag.retrieval.base import (
    Retriever as RetrieverBase,
)
from nanobot.rag.retrieval.bm25 import BM25Retriever
from nanobot.rag.retrieval.context import ContextExpander
from nanobot.rag.retrieval.hybrid import HybridRetriever
from nanobot.rag.retrieval.pipeline import AdvancedSearchPipeline
from nanobot.rag.retrieval.rerank import (
    CrossEncoderReranker,
    RerankService,
    SemanticDeduplicator,
)
from nanobot.rag.retrieval.vector import VectorRetriever

# ============================================================================
# Storage Layer
# ============================================================================
from nanobot.rag.storage.connection import DatabaseConnection
from nanobot.rag.storage.schema import (
    add_column_if_not_exists,
    create_fts_triggers,
    init_schema,
)
from nanobot.rag.store import DocumentStore

# ============================================================================
# All exports
# ============================================================================

__all__ = [
    # Data models
    "Chunk",
    "ChunkInfo",
    "Document",
    "DocumentInfo",
    "SearchResult",
    "SearchResultWithContext",
    "UnifiedSearchResult",
    "BaselineSearchResult",
    "BaselineChunkInfo",
    "BaselineDocumentInfo",
    # Core interfaces
    "AdvancedRetriever",
    "Chunker",
    "DocumentParser",
    "EmbeddingProvider",
    "Index",
    "Reranker",
    "Retriever",
    # Storage
    "DatabaseConnection",
    "init_schema",
    "create_fts_triggers",
    "add_column_if_not_exists",
    # Indexing layer
    "DocumentIndexer",
    "HierarchicalChunker",
    "HierarchicalChunks",
    "IndexingDocumentParser",
    "IndexingPipeline",
    "SemanticChunk",
    "SemanticChunker",
    # Retrieval layer
    "AdvancedRetrieverBase",
    "AdvancedSearchPipeline",
    "BM25Retriever",
    "ContextExpander",
    "CrossEncoderReranker",
    "HybridRetriever",
    "RerankService",
    "RetrieverBase",
    "SemanticDeduplicator",
    "VectorRetriever",
    # Other components
    "DocumentStore",
    "QueryExpander",
    "SearchCacheManager",
    "SentenceTransformerEmbeddingProvider",
    # Evaluation
    "DataGenerator",
    "EvalConfig",
    "EvalQuery",
    "EvalResult",
    "EvalSummary",
    "MetricsCalculator",
    "RAGEvaluator",
    "ResultJudge",
]
