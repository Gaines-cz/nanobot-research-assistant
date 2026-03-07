"""Core interfaces for the RAG module.

This module defines the abstract base classes that form the foundation
of the RAG system architecture.

Note: Retriever and AdvancedRetriever interfaces are now located in
nanobot.rag.retrieval.base to avoid duplication.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from nanobot.rag.models import (
    Chunk,
    Document,
    SearchResultWithContext,
)


class Index(ABC):
    """Abstract base class for document indexes."""

    @abstractmethod
    async def add_document(self, doc: Document, chunks: List[Chunk]) -> bool:
        """
        Add a document and its chunks to the index.

        Args:
            doc: The document to add
            chunks: The chunks of the document

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def delete_document(self, doc_id: int) -> bool:
        """
        Delete a document and its chunks from the index.

        Args:
            doc_id: The ID of the document to delete

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def update_document(self, doc: Document, chunks: List[Chunk]) -> bool:
        """
        Update a document and its chunks in the index.

        Args:
            doc: The updated document
            chunks: The updated chunks

        Returns:
            True if successful, False otherwise
        """
        pass


class DocumentParser(ABC):
    """Abstract base class for document parsers."""

    @abstractmethod
    def parse(self, path: Path) -> tuple[str, str]:
        """
        Parse a document and return its content and file type.

        Args:
            path: Path to the document

        Returns:
            Tuple of (content_text, file_type)
        """
        pass


class Chunker(ABC):
    """Abstract base class for text chunkers."""

    @abstractmethod
    def chunk(self, text: str) -> List[Chunk]:
        """
        Split text into chunks.

        Args:
            text: The text to split

        Returns:
            List of Chunk objects
        """
        pass


class Reranker(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        results: List[SearchResultWithContext]
    ) -> List[SearchResultWithContext]:
        """
        Rerank search results.

        Args:
            query: The original search query
            results: The initial search results

        Returns:
            Reranked list of SearchResultWithContext
        """
        pass


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Get the dimension size of the embeddings."""
        pass

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """
        Embed a single text string.

        Args:
            text: The text to embed

        Returns:
            The embedding vector
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple text strings.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass
