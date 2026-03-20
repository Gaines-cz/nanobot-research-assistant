"""Reranker and deduplicator for RAG search results.

This module provides reranking and deduplication functionality,
moved from the original rerank.py file.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from loguru import logger

from nanobot.config.schema import RAGConfig
from nanobot.utils.torch import get_optimal_device


class Reranker(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: list[str],
    ) -> list[tuple[int, float]]:
        """
        Rerank candidates against query.

        Args:
            query: Search query
            candidates: List of candidate texts to rerank

        Returns:
            List of (index, score) tuples sorted by score descending
        """
        pass


class CrossEncoderReranker(Reranker):
    """
    Cross-Encoder reranker optimized for MacBook Pro M4 24GB.

    Uses cross-encoder/ms-marco-MiniLM-L-6-v2:
    - Lightweight model (~80MB)
    - Fast inference on M4
    - Only reranks top-20 candidates for performance
    - Uses FP16 precision for faster inference and lower memory usage
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
        self._model_lock = asyncio.Lock()
        self._loaded = False

    async def _load_model(self) -> None:
        """Load CrossEncoder model lazily with thread safety."""
        if self._loaded:
            return

        async with self._model_lock:
            if self._loaded:
                return

            try:
                from sentence_transformers import CrossEncoder

                # 获取最优设备
                device = get_optimal_device()
                if device == "mps":
                    logger.info("MPS GPU acceleration enabled")

                logger.info("Loading CrossEncoder model: {} with FP16 on {}", self.model_name, device)

                # Run in thread pool to avoid blocking async loop
                loop = asyncio.get_event_loop()

                def _load_model():
                    model = CrossEncoder(
                        self.model_name,
                        max_length= 2048,
                        model_kwargs={
                            "torch_dtype": torch.float16,
                        }
                    )
                    # 手动移动模型到指定设备
                    model.model.to(device)
                    return model

                self._model = await loop.run_in_executor(None, _load_model)
                self._loaded = True
                logger.info("CrossEncoder model loaded successfully (FP16, {})", device.upper())
            except ImportError:
                logger.warning("sentence-transformers not available, rerank disabled")
                self._model = None
            except Exception as e:
                logger.warning("Could not load CrossEncoder model: {}", e)
                self._model = None

    async def rerank(
        self,
        query: str,
        candidates: list[str],
    ) -> list[tuple[int, float]]:
        """
        Rerank candidates using CrossEncoder.

        Args:
            query: Search query
            candidates: List of candidate texts

        Returns:
            List of (index, score) sorted by score descending.
            If reranker is not available, returns empty list.
        """
        if not candidates:
            return []

        await self._load_model()

        if self._model is None:
            return []

        try:
            # Prepare pairs: (query, candidate)
            pairs = [(query, candidate) for candidate in candidates]

            # Run inference in thread pool
            loop = asyncio.get_event_loop()

            rerank_start = time.perf_counter()
            scores = await loop.run_in_executor(
                None,
                lambda: self._model.predict(pairs)
            )
            rerank_elapsed = (time.perf_counter() - rerank_start) * 1000
            logger.info("[RAG PERF] CrossEncoder rerank: {} candidates, elapsed={:.1f}ms",
                        len(candidates), rerank_elapsed)

            # Create (index, score) tuples
            indexed_scores = [(i, float(score)) for i, score in enumerate(scores)]

            # Sort by score descending
            indexed_scores.sort(key=lambda x: x[1], reverse=True)

            return indexed_scores

        except Exception as e:
            logger.warning("CrossEncoder rerank failed: {}", e)
            return []


class SemanticDeduplicator:
    """
    Semantic deduplicator using embedding similarity.

    Removes chunks with cosine similarity >= threshold (default matches RAGDefaults.DEDUP_THRESHOLD).
    """

    def __init__(self, similarity_threshold: float = 0.7, config: Optional[RAGConfig] = None):
        # Backward compatibility: if config not provided, create one with the threshold
        if config is None:
            self.config = RAGConfig(dedup_threshold=similarity_threshold)
        else:
            self.config = config
        self.similarity_threshold = similarity_threshold

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        from nanobot.utils.helpers import cosine_similarity
        return cosine_similarity(a, b)

    async def deduplicate(
        self,
        chunks: list[str],
        embeddings: list[Optional[list[float]]],
    ) -> list[int]:
        """
        Deduplicate chunks using semantic similarity.

        Args:
            chunks: List of chunk texts
            embeddings: List of embeddings corresponding to chunks (can be None)

        Returns:
            List of indices to keep (chunks with similarity < threshold)
        """
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have same length")

        if not chunks:
            return []

        # Check if all embeddings are available
        if any(emb is None for emb in embeddings):
            # If any embedding is missing, return all indices (skip deduplication)
            return list(range(len(chunks)))

        # Use threshold from config if available
        threshold = self.config.dedup_threshold if self.config else self.similarity_threshold

        dedup_start = time.perf_counter()
        keep_indices: list[int] = []

        for i in range(len(chunks)):
            # Check if this chunk is too similar to any already kept chunk
            too_similar = False
            for j in keep_indices:
                # We already verified embeddings are not None above
                sim = self._cosine_similarity(embeddings[i], embeddings[j])  # type: ignore
                if sim >= threshold:
                    too_similar = True
                    break

            if not too_similar:
                keep_indices.append(i)

        dedup_elapsed = (time.perf_counter() - dedup_start) * 1000
        logger.info("[RAG PERF] Semantic dedup: {} -> {} chunks, elapsed={:.1f}ms",
                    len(chunks), len(keep_indices), dedup_elapsed)

        return keep_indices


class RerankService:
    """
    Combined reranking and deduplication service.

    Optimized for MacBook Pro M4 24GB:
    - Only reranks top-20 candidates
    - Applies rerank threshold (default matches RAGDefaults.RERANK_THRESHOLD)
    - Applies semantic deduplication (default matches RAGDefaults.DEDUP_THRESHOLD)

    NOTE: Default values here should match RAGDefaults in config/schema.py
    """

    def __init__(
        self,
        reranker: Optional[CrossEncoderReranker] = None,
        deduplicator: Optional[SemanticDeduplicator] = None,
        config: Optional[RAGConfig] = None,
        # Backward compatibility parameters
        rerank_threshold: Optional[float] = None,
        dedup_threshold: Optional[float] = None,
        rerank_top_k: Optional[int] = None,
        enable_rerank: Optional[bool] = None,
    ):
        # Backward compatibility: if old parameters are provided, create config from them
        if config is None and (rerank_threshold is not None or dedup_threshold is not None or
                                 rerank_top_k is not None or enable_rerank is not None):
            config = RAGConfig()
            if rerank_threshold is not None:
                config.rerank_threshold = rerank_threshold
            if dedup_threshold is not None:
                config.dedup_threshold = dedup_threshold
            if rerank_top_k is not None:
                config.rerank_top_k = rerank_top_k
            if enable_rerank is not None:
                config.enable_rerank = enable_rerank

        self.config = config or RAGConfig()
        self.reranker = reranker or CrossEncoderReranker(self.config.rerank_model)
        self.deduplicator = deduplicator or SemanticDeduplicator(self.config.dedup_threshold, self.config)

    async def rerank_and_dedup(
        self,
        query: str,
        candidates: list[tuple[str, Any, Optional[list[float]]]],
    ) -> list[tuple[int, Any, float]]:
        """
        Apply reranking and deduplication.

        Args:
            query: Search query
            candidates: List of (text, metadata, embedding) tuples

        Returns:
            List of (original_index, metadata, final_score) sorted by final_score,
            filtered by rerank threshold and deduplicated.
        """
        if not candidates:
            return []

        # Extract texts and prepare indices
        candidate_texts = [c[0] for c in candidates]
        candidate_metadatas = [c[1] for c in candidates]
        candidate_embeddings = [c[2] for c in candidates]

        # Step 1: Rerank (only top-rerank_top_k for M4 optimization)
        reranked_indices: list[tuple[int, float]] = []
        if self.config.enable_rerank and len(candidates) > 0:
            # Take top candidates to rerank (already sorted by score from merge step)
            # 同时保存原始索引: (original_idx, text)
            # 先 enumerate 再切片，确保保存的是原始索引
            rerank_candidates_with_idx = [
                (i, text)
                for i, text in enumerate(candidate_texts)
            ][:self.config.rerank_top_k]
            rerank_candidates = [text for _, text in rerank_candidates_with_idx]
            rerank_results = await self.reranker.rerank(query, rerank_candidates)

            if rerank_results:
                # Map back to original indices with scores
                reranked_indices = []
                for rerank_idx, score in rerank_results:
                    original_idx = rerank_candidates_with_idx[rerank_idx][0]
                    reranked_indices.append((original_idx, score))

        # If no rerank results or disabled, use original order with dummy scores
        if not reranked_indices:
            reranked_indices = [(i, 1.0 - (i * 0.01)) for i in range(len(candidates))]

        # Step 2: Filter by rerank threshold
        thresholded = [
            (idx, meta, score)
            for idx, score in reranked_indices
            for meta in [candidate_metadatas[idx]]
            if score >= self.config.rerank_threshold
        ]

        # If everything filtered out, keep top 3 without threshold
        if not thresholded and reranked_indices:
            thresholded = [
                (idx, candidate_metadatas[idx], score)
                for idx, score in reranked_indices[:3]
            ]

        # Step 3: Semantic deduplication (only if all embeddings are available)
        # Skip semantic deduplication if any embedding is missing (e.g., large chunks in dual mode)
        if len(thresholded) > 1:
            # Get embeddings for thresholded candidates
            dedup_indices = [t[0] for t in thresholded]
            dedup_embeddings = [candidate_embeddings[i] for i in dedup_indices]
            dedup_texts = [candidate_texts[i] for i in dedup_indices]

            # Check if all embeddings are available
            all_embeddings_available = all(emb is not None for emb in dedup_embeddings)

            if all_embeddings_available:
                # Deduplicate
                keep_in_thresholded = await self.deduplicator.deduplicate(
                    dedup_texts, dedup_embeddings
                )

                # Keep only deduplicated entries
                thresholded = [thresholded[i] for i in keep_in_thresholded]
            else:
                # Skip semantic deduplication when embeddings are missing
                logger.debug("Skipping semantic deduplication: some embeddings are missing")

        # Sort by final score
        thresholded.sort(key=lambda x: x[2], reverse=True)

        return thresholded
