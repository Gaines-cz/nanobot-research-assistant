"""Embedding providers for RAG using sentence-transformers."""

from abc import ABC, abstractmethod
from threading import Lock
from typing import Any

from loguru import logger


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings."""
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        pass


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using sentence-transformers (local, no API required)."""

    _model_cache: dict[str, Any] = {}
    _model_lock = Lock()

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        """
        Initialize with a sentence-transformers model.

        Common models:
        - all-MiniLM-L6-v2 (default, fast, 384 dim)
        - all-mpnet-base-v2 (better quality, 768 dim)
        - paraphrase-multilingual-MiniLM-L12-v2 (multilingual)
        """
        self.model_name = model
        self._model: Any | None = None
        self._dimensions: int | None = None

    def _load_model(self) -> None:
        """Load the sentence-transformers model (lazy loading)."""
        if self._model is not None:
            return

        with SentenceTransformerEmbeddingProvider._model_lock:
            # Check again after acquiring lock
            if self._model is not None:
                return

            # Check cache
            if self.model_name in SentenceTransformerEmbeddingProvider._model_cache:
                logger.debug("Using cached sentence-transformers model: {}", self.model_name)
                self._model = SentenceTransformerEmbeddingProvider._model_cache[self.model_name]
            else:
                logger.info("Loading sentence-transformers model: {}", self.model_name)
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError:
                    raise ImportError(
                        "sentence-transformers is required. "
                        "Install with: pip install 'nanobot-ai[rag]'"
                    )
                self._model = SentenceTransformer(self.model_name)
                SentenceTransformerEmbeddingProvider._model_cache[self.model_name] = self._model

            # Get embedding dimensions by doing a test embedding
            test_embedding = self._model.encode("test")
            self._dimensions = len(test_embedding)
            logger.debug("Embedding dimensions: {}", self._dimensions)

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        result = await self.embed_batch([text])
        return result[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings."""
        import asyncio

        self._load_model()
        assert self._model is not None

        # Truncate texts to avoid OOM (bge-m3 has ~8192 token limit)
        # We truncate at ~4000 chars as a safe guard
        max_text_chars = 4000
        truncated_texts = [text[:max_text_chars] if len(text) > max_text_chars else text for text in texts]

        # Run in thread pool to avoid blocking event loop
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(None, self._model.encode, truncated_texts)
        return [e.tolist() for e in embeddings]

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        if self._dimensions is None:
            self._load_model()
        assert self._dimensions is not None
        return self._dimensions
