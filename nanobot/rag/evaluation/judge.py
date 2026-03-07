"""RAG Evaluation - Result judge (Hybrid mode)."""

import math
from typing import List, Optional, Tuple

from loguru import logger

from nanobot.rag.evaluation.base import EvalQuery
from nanobot.rag.models import SearchResultWithContext


class ResultJudge:
    """
    Result judge with Hybrid mode.

    Priority:
    1. ID match (most reliable) - uses source_chunk_id from test data generation
    2. Parent ID match (for dual granularity) - if result's parent is the target
    3. Semantic similarity - uses cosine similarity between golden_embedding and result embeddings
    """

    def __init__(
        self,
        strong_threshold: float = 0.7,
        weak_threshold: float = 0.6,
        db_connection=None,
    ):
        self.strong_threshold = strong_threshold
        self.weak_threshold = weak_threshold
        self._db = db_connection
        # Cache for parent chunk lookups
        self._parent_cache: dict[int, Optional[int]] = {}
        # Cache for child lookups (reverse parent-child)
        self._child_cache: dict[str, set[int]] = {}

    def judge(
        self,
        results: List[SearchResultWithContext],
        query: EvalQuery,
        golden_embedding: Optional[List[float]] = None,
    ) -> Tuple[bool, Optional[str], Optional[str], Optional[float]]:
        """
        Judge if search results pass (Hybrid mode).

        Priority:
        1. ID match (most reliable)
        2. Parent ID match (for dual granularity)
        3. Semantic similarity (if golden_embedding provided)

        Args:
            results: List of search results
            query: Test query
            golden_embedding: Precomputed embedding of golden_context

        Returns:
            (hit, hit_reason, failure_reason, best_similarity)
            hit_reason: "id_match_rank_{n}" / "parent_id_match_rank_{n}" / "strong_semantic_rank_{n}_sim_{x.xx}" / "weak_semantic_rank_{n}_sim_{x.xx}"
            failure_reason: "recall_failed" / "low_similarity"
            best_similarity: Highest similarity score found (if any)
        """
        best_similarity: Optional[float] = None

        # 1. First try ID match (highest priority)
        hit, reason = self._judge_by_id(results, query)
        if hit:
            # Calculate best similarity even for ID matches
            if golden_embedding is not None and results:
                _, best_sim = self._calculate_best_similarity(results, golden_embedding)
                best_similarity = best_sim
            return hit, reason, None, best_similarity

        # 2. Try parent ID match (for dual granularity)
        hit, reason = self._judge_by_parent_id(results, query)
        if hit:
            if golden_embedding is not None and results:
                _, best_sim = self._calculate_best_similarity(results, golden_embedding)
                best_similarity = best_sim
            return hit, reason, None, best_similarity

        # 3. Try semantic similarity
        if golden_embedding is not None and results:
            hit, reason, best_sim = self._judge_by_semantic_similarity(
                results, golden_embedding
            )
            best_similarity = best_sim
            if hit:
                return hit, reason, None, best_similarity

        # 4. No match found, but still return best similarity if available
        if golden_embedding is not None and results:
            _, best_sim = self._calculate_best_similarity(results, golden_embedding)
            best_similarity = best_sim

        return False, None, "recall_failed", best_similarity

    def _judge_by_id(
        self,
        results: List[SearchResultWithContext],
        query: EvalQuery,
    ) -> Tuple[bool, Optional[str]]:
        """Judge by ID match."""
        if query.source_chunk_id is None:
            return False, None

        for i, result in enumerate(results):
            if result.chunk.id == query.source_chunk_id:
                return True, f"id_match_rank_{i+1}"

        return False, None

    def _judge_by_parent_id(
        self,
        results: List[SearchResultWithContext],
        query: EvalQuery,
    ) -> Tuple[bool, Optional[str]]:
        """
        Judge by parent ID match (for dual granularity).

        Checks two scenarios:
        1. Result's parent_id matches query's source_chunk_id (forward)
        2. Query's source_chunk_id has children that match result's chunk_id (reverse)
        """
        if query.source_chunk_id is None:
            return False, None

        if not self._db:
            return False, None

        for i, result in enumerate(results):
            # Scenario 1: Forward - result's parent is the target chunk
            parent_id = self._get_parent_chunk_id(result.chunk.id)
            if parent_id == query.source_chunk_id:
                logger.debug("[DualGranularity] Parent ID match (forward): result={}, parent={}, target={}",
                            result.chunk.id, parent_id, query.source_chunk_id)
                return True, f"parent_id_match_rank_{i+1}"

            # Scenario 2: Reverse - result is a child of the target chunk
            is_child = self._is_child_of(result.chunk.id, query.source_chunk_id)
            if is_child:
                logger.debug("[DualGranularity] Parent ID match (reverse): result={}, target_parent={}",
                            result.chunk.id, query.source_chunk_id)
                return True, f"parent_id_match_rank_{i+1}"

        return False, None

    def _is_child_of(self, chunk_id: int, parent_id: int) -> bool:
        """Check if chunk_id is a child of parent_id (reverse lookup)."""
        cache_key = f"child_of_{parent_id}"
        if cache_key in self._child_cache:
            child_ids = self._child_cache[cache_key]
            return chunk_id in child_ids

        if not self._db:
            return False

        try:
            cursor = self._db.db.execute("""
                SELECT id FROM chunks WHERE parent_chunk_id = ?
            """, (parent_id,))
            rows = cursor.fetchall()
            child_ids = {row[0] for row in rows}

            # Cache the result
            self._child_cache[cache_key] = child_ids

            return chunk_id in child_ids
        except Exception as e:
            logger.debug("[DualGranularity] Failed to check child relationship: {}", e)
            return False

    def _get_parent_chunk_id(self, chunk_id: int) -> Optional[int]:
        """Get parent chunk ID from database (with caching)."""
        if chunk_id in self._parent_cache:
            return self._parent_cache[chunk_id]

        if not self._db:
            return None

        try:
            cursor = self._db.db.execute("""
                SELECT parent_chunk_id FROM chunks WHERE id = ?
            """, (chunk_id,))
            row = cursor.fetchone()
            parent_id = row[0] if row else None
            self._parent_cache[chunk_id] = parent_id
            return parent_id
        except Exception as e:
            logger.debug("[DualGranularity] Failed to get parent chunk ID: {}", e)
            self._parent_cache[chunk_id] = None
            return None

    def _judge_by_semantic_similarity(
        self,
        results: List[SearchResultWithContext],
        golden_embedding: List[float],
    ) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Judge by semantic similarity between golden context and results.

        Uses cosine similarity between embeddings.

        Returns:
            (hit, reason, best_similarity)
        """
        best_sim: Optional[float] = None

        for i, result in enumerate(results):
            # Check if result has embedding
            if hasattr(result.chunk, 'embedding') and result.chunk.embedding:
                sim = self._cosine_sim(golden_embedding, result.chunk.embedding)

                # Track best similarity
                if best_sim is None or sim > best_sim:
                    best_sim = sim

                if sim >= self.strong_threshold:
                    return True, f"strong_semantic_rank_{i+1}_sim_{sim:.2f}", best_sim
                elif sim >= self.weak_threshold:
                    return True, f"weak_semantic_rank_{i+1}_sim_{sim:.2f}", best_sim

        return False, None, best_sim

    def _calculate_best_similarity(
        self,
        results: List[SearchResultWithContext],
        golden_embedding: List[float],
    ) -> Tuple[Optional[int], Optional[float]]:
        """
        Calculate the best similarity score among all results.

        Returns:
            (best_rank, best_similarity) - both None if no embeddings available
        """
        best_sim: Optional[float] = None
        best_rank: Optional[int] = None

        for i, result in enumerate(results):
            if hasattr(result.chunk, 'embedding') and result.chunk.embedding:
                sim = self._cosine_sim(golden_embedding, result.chunk.embedding)
                if best_sim is None or sim > best_sim:
                    best_sim = sim
                    best_rank = i + 1

        return best_rank, best_sim

    def _cosine_sim(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)
