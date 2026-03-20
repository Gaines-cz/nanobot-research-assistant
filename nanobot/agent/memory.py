"""
MemoryStore with hybrid search using SQLite + RAG components.

This module provides the memory system with:
- SQLite-backed storage via MemoryDatabase
- Hybrid search: BM25 + Vector + RRF fusion + CrossEncoder reranking
- 2-step consolidation flow for memory evolution
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from loguru import logger

from nanobot.agent.memory_db import MemoryDatabase
from nanobot.rag.embeddings import EmbeddingProvider
from nanobot.rag.retrieval.rerank import CrossEncoderReranker

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session


class MemoryType(Enum):
    """Memory type enumeration."""
    HISTORY = "HISTORY"
    KNOWLEDGE = "KNOWLEDGE"
    DECISIONS = "DECISIONS"
    PROJECTS = "PROJECTS"


# Constants for hybrid search
BM25_TOP_K = 50
VECTOR_TOP_K = 50
RERANK_TOP_K = 20
RECALL_TOP_K = 50

# Scoring weights for comprehensive ranking
MODEL_SCORE_WEIGHT = 0.7
FREQ_SCORE_WEIGHT = 0.2
RECENCY_SCORE_WEIGHT = 0.1

# Max read times for normalization
MAX_READ_TIMES = 100

# Recency decay lambda (days)
RECENCY_DECAY_LAMBDA = 0.1


@dataclass
class MemorySearchResult:
    """Memory search result with metadata."""
    id: int
    type: str
    detail: str
    at_time: int
    read_times: int
    last_read_time: int
    model_score: float = 0.0
    final_score: float = 0.0
    source: str = ""  # "bm25", "vector", "rrf"


class MemoryStore:
    """
    SQLite-backed memory store with hybrid search.

    Features:
    - SQLite storage via MemoryDatabase
    - Hybrid search: BM25 + Vector + RRF fusion + CrossEncoder reranking
    - Comprehensive scoring: model_score * 0.7 + freq_score * 0.2 + recency_score * 0.1
    - 2-step consolidation for memory evolution
    """

    def __init__(
        self,
        workspace: Path,
        embedding_provider: Optional[EmbeddingProvider] = None,
        memory_db_path: Optional[Path] = None,
    ):
        """
        Initialize MemoryStore.

        Args:
            workspace: Workspace path for memory storage
            embedding_provider: Optional embedding provider for vector search
            memory_db_path: Optional custom path for memory database
        """
        self._workspace = workspace
        self._embedding_provider = embedding_provider

        # Initialize memory database
        if memory_db_path:
            self._db_path = memory_db_path
        else:
            self._db_path = workspace / "memory" / "memory.db"
        self._db = MemoryDatabase(self._db_path)

        # Initialize memory-specific FTS5 table for BM25 search
        self._init_memory_fts()

        # Initialize memory-specific vector storage
        self._vec_enabled = False
        self._init_memory_vectors()

        # Reranker for hybrid search
        self._reranker = CrossEncoderReranker()

        logger.info(
            "[MemoryStore] Initialized with hybrid search: BM25 + Vector (vec_enabled={})",
            self._vec_enabled
        )

    def _init_memory_fts(self) -> None:
        """Initialize memory-specific FTS5 table for BM25 search."""
        try:
            conn = self._db._conn
            # Create FTS5 virtual table for memory content search
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    detail,
                    content=memories,
                    content_rowid=id,
                    tokenize='porter unicode61'
                )
            """)
            # Create triggers for automatic index updates
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, detail) VALUES (new.id, new.detail);
                END
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, detail) VALUES('delete', old.id, old.detail);
                END
            """)
            conn.commit()
            logger.debug("[MemoryStore] FTS5 table for memories initialized")
        except Exception as e:
            logger.warning("[MemoryStore] Failed to initialize FTS5: {}", e)

    def _init_memory_vectors(self) -> None:
        """Initialize memory-specific vector storage using sqlite-vec."""
        if self._embedding_provider is None:
            logger.debug("[MemoryStore] No embedding provider, vector search disabled")
            return

        try:
            import sqlite_vec

            conn = self._db._conn

            # Enable extension loading and load sqlite-vec
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)

            # Get embedding dimensions
            dims = self._embedding_provider.dimensions

            # Create memory embeddings table
            conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_embeddings USING vec0(
                    memory_id INTEGER PRIMARY KEY,
                    embedding FLOAT32[{dims}]
                )
            """)
            conn.commit()

            self._vec_enabled = True
            logger.info("[MemoryStore] Vector storage enabled (dims={})", dims)
        except Exception as e:
            logger.warning("[MemoryStore] Vector storage disabled: {}", e)
            self._vec_enabled = False

    async def _bm25_search(self, query: str, type_filter: Optional[str] = None, top_k: int = BM25_TOP_K) -> list[MemorySearchResult]:
        """
        Perform BM25 full-text search on memories.

        Args:
            query: Search query
            type_filter: Optional memory type filter
            top_k: Number of results to return

        Returns:
            List of MemorySearchResult sorted by BM25 score
        """
        if not query.strip():
            return []

        try:
            # Sanitize query for FTS5
            safe_query = self._sanitize_fts_query(query)
            if not safe_query:
                return []

            conn = self._db._conn

            if type_filter:
                # Join with memories table to filter by type
                sql = """
                    SELECT
                        m.id,
                        m.type,
                        m.detail,
                        m.at_time,
                        m.read_times,
                        m.last_read_time,
                        bm25(memories_fts) as score
                    FROM memories_fts
                    JOIN memories m ON memories_fts.rowid = m.id
                    WHERE memories_fts MATCH ?
                      AND m.type = ?
                      AND m.deleted_at IS NULL
                    ORDER BY bm25(memories_fts)
                    LIMIT ?
                """
                cursor = conn.execute(sql, (safe_query, type_filter, top_k))
            else:
                sql = """
                    SELECT
                        m.id,
                        m.type,
                        m.detail,
                        m.at_time,
                        m.read_times,
                        m.last_read_time,
                        bm25(memories_fts) as score
                    FROM memories_fts
                    JOIN memories m ON memories_fts.rowid = m.id
                    WHERE memories_fts MATCH ?
                      AND m.deleted_at IS NULL
                    ORDER BY bm25(memories_fts)
                    LIMIT ?
                """
                cursor = conn.execute(sql, (safe_query, top_k))

            results = []
            rows = cursor.fetchall()
            if rows:
                # Min-Max normalization for BM25 scores (lower BM25 = better)
                bm25_scores = [row[6] for row in rows if row[6] is not None]
                if bm25_scores:
                    min_bm25 = min(bm25_scores)
                    max_bm25 = max(bm25_scores)

                    for row in rows:
                        bm25_score = row[6] if row[6] is not None else 1.0
                        if max_bm25 == min_bm25:
                            normalized = 1.0
                        else:
                            # Invert because BM25: lower = better
                            normalized = 1.0 - (bm25_score - min_bm25) / (max_bm25 - min_bm25)

                        results.append(MemorySearchResult(
                            id=row[0],
                            type=row[1],
                            detail=row[2],
                            at_time=row[3],
                            read_times=row[4],
                            last_read_time=row[5] if row[5] else row[3],
                            model_score=normalized,
                            source="bm25",
                        ))

            logger.debug("[MemoryStore] BM25 search: query='{}', results={}", query[:50], len(results))
            return results

        except Exception as e:
            logger.warning("[MemoryStore] BM25 search failed: {}", e)
            return []

    async def _vector_search(self, query: str, type_filter: Optional[str] = None, top_k: int = VECTOR_TOP_K) -> list[MemorySearchResult]:
        """
        Perform vector similarity search on memories.

        Args:
            query: Search query
            type_filter: Optional memory type filter
            top_k: Number of results to return

        Returns:
            List of MemorySearchResult sorted by similarity
        """
        if not self._vec_enabled or self._embedding_provider is None:
            return []

        try:
            import sqlite_vec

            # Generate query embedding
            query_embedding = await self._embedding_provider.embed(query)
            embedding_blob = sqlite_vec.serialize_float32(query_embedding)

            conn = self._db._conn

            if type_filter:
                sql = """
                    SELECT
                        m.id,
                        m.type,
                        m.detail,
                        m.at_time,
                        m.read_times,
                        m.last_read_time,
                        e.distance
                    FROM memory_embeddings e
                    JOIN memories m ON e.memory_id = m.id
                    WHERE e.embedding MATCH ?
                      AND e.k = ?
                      AND m.type = ?
                      AND m.deleted_at IS NULL
                """
                cursor = conn.execute(sql, (embedding_blob, top_k, type_filter))
            else:
                sql = """
                    SELECT
                        m.id,
                        m.type,
                        m.detail,
                        m.at_time,
                        m.read_times,
                        m.last_read_time,
                        e.distance
                    FROM memory_embeddings e
                    JOIN memories m ON e.memory_id = m.id
                    WHERE e.embedding MATCH ?
                      AND e.k = ?
                      AND m.deleted_at IS NULL
                """
                cursor = conn.execute(sql, (embedding_blob, top_k))

            results = []
            for row in cursor:
                distance = row[6] if row[6] is not None else 1.0
                similarity = max(0.0, min(1.0, 1.0 - (distance / 2.0)))

                results.append(MemorySearchResult(
                    id=row[0],
                    type=row[1],
                    detail=row[2],
                    at_time=row[3],
                    read_times=row[4],
                    last_read_time=row[5] if row[5] else row[3],
                    model_score=similarity,
                    source="vector",
                ))

            logger.debug("[MemoryStore] Vector search: query='{}', results={}", query[:50], len(results))
            return results

        except Exception as e:
            logger.warning("[MemoryStore] Vector search failed: {}", e)
            return []

    def _rrf_fusion(self, bm25_results: list[MemorySearchResult], vector_results: list[MemorySearchResult], k: int = 60) -> list[MemorySearchResult]:
        """
        Perform Reciprocal Rank Fusion (RRF) on BM25 and vector results.

        Args:
            bm25_results: BM25 search results
            vector_results: Vector search results
            k: RRF smoothing parameter

        Returns:
            Fused and deduplicated results
        """
        seen = {}
        fused: list[MemorySearchResult] = []

        # Process BM25 results first (higher priority)
        for rank, result in enumerate(bm25_results):
            if result.id not in seen:
                rrf_score = 1.0 / (k + rank + 1)
                result.model_score = rrf_score
                result.source = "bm25"
                seen[result.id] = result
                fused.append(result)

        # Add vector results not in BM25
        for rank, result in enumerate(vector_results):
            if result.id not in seen:
                rrf_score = 1.0 / (k + rank + 1)
                result.model_score = rrf_score
                result.source = "vector"
                seen[result.id] = result
                fused.append(result)

        # Sort by RRF score
        fused.sort(key=lambda x: x.model_score, reverse=True)

        return fused

    async def _rerank_with_cross_encoder(self, query: str, candidates: list[MemorySearchResult], top_k: int = RERANK_TOP_K) -> list[MemorySearchResult]:
        """
        Rerank candidates using CrossEncoder.

        Args:
            query: Search query
            candidates: Candidate results to rerank
            top_k: Number of results to return after reranking

        Returns:
            Reranked results
        """
        if not candidates:
            return []

        try:
            # Prepare candidate texts
            candidate_texts = [c.detail for c in candidates]

            # Get reranked indices with scores
            reranked = await self._reranker.rerank(query, candidate_texts)

            if not reranked:
                return candidates[:top_k]

            # Map back to MemorySearchResult with cross-encoder scores
            reranked_results = []
            for idx, ce_score in reranked[:top_k]:
                result = candidates[idx]
                result.model_score = ce_score
                reranked_results.append(result)

            logger.debug("[MemoryStore] CrossEncoder rerank: {} -> {}", len(candidates), len(reranked_results))
            return reranked_results

        except Exception as e:
            logger.warning("[MemoryStore] CrossEncoder rerank failed: {}", e)
            return candidates[:top_k]

    def _calculate_final_score(self, result: MemorySearchResult) -> float:
        """
        Calculate comprehensive final score.

        Formula: final = model_score * 0.7 + freq_score * 0.2 + recency_score * 0.1

        Args:
            result: MemorySearchResult with model_score

        Returns:
            Final comprehensive score
        """
        # Frequency score: log normalization
        freq_score = math.log(1 + result.read_times) / math.log(1 + MAX_READ_TIMES)

        # Recency score: exponential decay
        days_since = (time.time() - result.last_read_time) / 86400
        recency_score = math.exp(-RECENCY_DECAY_LAMBDA * days_since)

        # Weighted sum
        final_score = (
            result.model_score * MODEL_SCORE_WEIGHT +
            freq_score * FREQ_SCORE_WEIGHT +
            recency_score * RECENCY_SCORE_WEIGHT
        )

        return final_score

    async def search(
        self,
        query: str,
        type: Optional[MemoryType] = None,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Hybrid search for memories.

        Flow: Query -> BM25 search -> Vector search -> RRF fusion -> CrossEncoder rerank -> 综合打分 -> Top-K

        Args:
            query: Search query
            type: Optional memory type filter
            top_k: Number of results to return

        Returns:
            List of memory dicts sorted by comprehensive score
        """
        if not query.strip():
            return []

        type_filter = type.value if type else None

        # Step 1: BM25 search
        bm25_start = time.perf_counter()
        bm25_results = await self._bm25_search(query, type_filter, RECALL_TOP_K)
        bm25_elapsed = (time.perf_counter() - bm25_start) * 1000

        # Step 2: Vector search
        vector_start = time.perf_counter()
        vector_results = await self._vector_search(query, type_filter, RECALL_TOP_K)
        vector_elapsed = (time.perf_counter() - vector_start) * 1000

        # Step 3: RRF fusion
        fusion_start = time.perf_counter()
        fused_results = self._rrf_fusion(bm25_results, vector_results)
        fusion_elapsed = (time.perf_counter() - fusion_start) * 1000

        # If no vector results, fall back to BM25-only
        if not vector_results:
            fused_results = bm25_results[:top_k]

        # Step 4: CrossEncoder reranking (on top candidates)
        rerank_start = time.perf_counter()
        reranked_results = await self._rerank_with_cross_encoder(query, fused_results, RERANK_TOP_K)
        rerank_elapsed = (time.perf_counter() - rerank_start) * 1000

        # Step 5: Calculate final scores and sort
        for result in reranked_results:
            result.final_score = self._calculate_final_score(result)

        reranked_results.sort(key=lambda x: x.final_score, reverse=True)

        # Step 6: Take top-k and update read stats
        final_results = reranked_results[:top_k]

        # Update read statistics (direct sync call in same thread)
        for result in final_results:
            self._db.update_read_stats(result.id)

        logger.info(
            "[MemoryStore] Search: query='{}', bm25={}ms, vec={}ms, fusion={}ms, rerank={}ms, final={}",
            query[:50], bm25_elapsed, vector_elapsed, fusion_elapsed, rerank_elapsed, len(final_results)
        )

        return [
            {
                "id": r.id,
                "type": r.type,
                "detail": r.detail,
                "at_time": r.at_time,
                "read_times": r.read_times,
                "last_read_time": r.last_read_time,
                "score": r.final_score,
            }
            for r in final_results
        ]

    async def insert(self, type: MemoryType, detail: str, at_time: Optional[int] = None, read_times: int = 0) -> int:
        """
        Insert a new memory.

        Args:
            type: Memory type
            detail: Memory content (Markdown format)
            at_time: Optional associated timestamp (defaults to now)
            read_times: Initial read times (default 0)

        Returns:
            Generated memory id (INTEGER)
        """
        if at_time is None:
            at_time = int(time.time())

        memory_id = self._db.insert(type.value, detail, at_time, read_times)

        # Store vector embedding if enabled
        if self._vec_enabled and self._embedding_provider:
            try:
                import sqlite_vec
                embedding = await self._embedding_provider.embed(detail)
                embedding_blob = sqlite_vec.serialize_float32(embedding)

                conn = self._db._conn
                conn.execute(
                    "INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding) VALUES (?, ?)",
                    (memory_id, embedding_blob)
                )
                conn.commit()
            except Exception as e:
                logger.warning("[MemoryStore] Failed to store vector embedding: {}", e)

        logger.debug("[MemoryStore] Inserted memory: id={}, type={}", memory_id, type.value)
        return memory_id

    async def update(self, id: int, detail: str, at_time: int) -> bool:
        """
        Update a memory.

        Args:
            id: Memory id
            detail: New memory content
            at_time: New associated timestamp

        Returns:
            True if updated, False if not found
        """
        # Get old detail for FTS sync before update
        conn = self._db._conn
        cursor = conn.execute("SELECT detail FROM memories WHERE id = ? AND deleted_at IS NULL", (id,))
        row = cursor.fetchone()
        if not row:
            return False

        old_detail = row[0]
        now = self._db._now()
        sql = """
        UPDATE memories
        SET detail = ?, at_time = ?, updated_at = ?
        WHERE id = ? AND deleted_at IS NULL
        """
        cursor = conn.execute(sql, (detail, at_time, now, id))
        conn.commit()
        updated = cursor.rowcount > 0

        if updated:
            # Sync FTS: mark old as deleted, insert new
            conn.execute("INSERT INTO memories_fts(memories_fts, rowid, detail) VALUES('delete', ?, ?)", (id, old_detail))
            conn.execute("INSERT INTO memories_fts(rowid, detail) VALUES (?, ?)", (id, detail))
            conn.commit()

        # Update vector embedding if enabled
        if updated and self._vec_enabled and self._embedding_provider:
            try:
                import sqlite_vec
                embedding = await self._embedding_provider.embed(detail)
                embedding_blob = sqlite_vec.serialize_float32(embedding)

                conn = self._db._conn
                conn.execute(
                    "INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding) VALUES (?, ?)",
                    (id, embedding_blob)
                )
                conn.commit()
            except Exception as e:
                logger.warning("[MemoryStore] Failed to update vector embedding: {}", e)

        if updated:
            logger.debug("[MemoryStore] Updated memory: id={}", id)
        return updated

    async def delete(self, id: int) -> bool:
        """
        Soft delete a memory.

        Args:
            id: Memory id

        Returns:
            True if deleted, False if not found
        """
        deleted = self._db.soft_delete(id)
        if deleted:
            logger.debug("[MemoryStore] Soft deleted memory: id={}", id)
        return deleted

    def purge_candidates(
        self,
        memory_type: Optional[MemoryType] = None,
        ratio: float = 5.0,
    ) -> list[dict]:
        """
        Get memory candidates for purge (LFU-based).

        Args:
            memory_type: Optional specific type to purge. If None, returns candidates for all types.
            ratio: Purge ratio percentage (1-5, default 5)

        Returns:
            List of memory dicts marked for purge, grouped by type
        """
        if ratio < 1:
            ratio = 1
        elif ratio > 5:
            ratio = 5

        conn = self._db._conn
        results: dict[str, list[dict]] = {}

        types_to_purge = [memory_type] if memory_type else list(MemoryType)

        for mtype in types_to_purge:
            # Get all non-deleted memories of this type, sorted by LFU
            # LFU: lower read_times first, then older last_read_time first
            cursor = conn.execute("""
                SELECT id, type, detail, at_time, read_times, last_read_time
                FROM memories
                WHERE type = ? AND deleted_at IS NULL
                ORDER BY read_times ASC, last_read_time ASC
            """, (mtype.value,))

            rows = cursor.fetchall()
            if not rows:
                continue

            # Calculate purge count: min(count * ratio%, 5% of total)
            # "5% of total" means at most 5% of total memories can be purged in one run
            total = len(rows)
            purge_count = max(1, int(total * ratio / 100))
            purge_count = min(purge_count, max(1, int(total * 0.05)))

            candidates = []
            for row in rows[:purge_count]:
                candidates.append({
                    "id": row[0],
                    "type": row[1],
                    "detail": row[2],
                    "at_time": row[3],
                    "read_times": row[4],
                    "last_read_time": row[5],
                })

            results[mtype.value] = candidates

        return results

    def purge(
        self,
        memory_type: Optional[MemoryType] = None,
        ratio: float = 5.0,
    ) -> dict[str, int]:
        """
        Execute purge on memories (LFU-based).

        Args:
            memory_type: Optional specific type to purge. If None, purges all types.
            ratio: Purge ratio percentage (1-5, default 5)

        Returns:
            Dict mapping type to number of purged memories
        """
        candidates = self.purge_candidates(memory_type, ratio)

        purged_counts: dict[str, int] = {}
        for mtype, items in candidates.items():
            count = 0
            for item in items:
                if self._db.hard_delete(item["id"]):
                    count += 1
                    logger.debug("[MemoryStore] Purged memory: id={}, type={}", item["id"], mtype)
            if count > 0:
                purged_counts[mtype] = count
                logger.info("[MemoryStore] Purged {} {} memories", count, mtype)

        return purged_counts

    async def consolidate(self, session: "Session", provider: "LLMProvider", model: str, **kwargs) -> bool:
        """
        Consolidate memory from session using 2-step flow.

        Step 1: Extract memories (1 LLM call)
        Step 2: Process by type:
          - HISTORY -> INSERT
          - KNOWLEDGE/DECISIONS/PROJECTS -> search -> INSERT/UPDATE

        Args:
            session: Session with messages to consolidate
            provider: LLM provider for extraction
            model: Model name for extraction

        Returns:
            True on success, False on failure
        """
        try:

            # Get messages since last consolidation
            messages = session.messages[session.last_consolidated:]
            if not messages:
                return True

            # Build conversation text from all unconsolidated messages
            conversation = self._build_conversation_text(messages)

            # Step 1: Extract memories with LLM
            extraction_start = time.perf_counter()
            extracted = await self._extract_memories(conversation, provider, model)
            extraction_elapsed = (time.perf_counter() - extraction_start) * 1000

            if not extracted:
                logger.info("[MemoryStore] No memories extracted from session")
                return True

            logger.info(
                "[MemoryStore] Extracted memories: history={}, knowledge={}, decisions={}, projects={}, elapsed={}ms",
                len(extracted.get("history", [])), len(extracted.get("knowledge", [])),
                len(extracted.get("decisions", [])), len(extracted.get("projects", [])),
                extraction_elapsed
            )

            # Step 2: Process by type
            process_start = time.perf_counter()

            # 2.1 HISTORY - direct INSERT
            if extracted.get("history"):
                history_text = extracted["history"]
                if isinstance(history_text, list):
                    history_text = "\n".join(history_text)
                await self.insert(MemoryType.HISTORY, history_text)

            # 2.2 KNOWLEDGE - search then INSERT or UPDATE
            for knowledge in extracted.get("knowledge", []):
                if isinstance(knowledge, dict):
                    content = knowledge.get("content", str(knowledge))
                    title = knowledge.get("title", "")
                    if title:
                        content = f"## {title}\n\n{content}"
                else:
                    content = str(knowledge)

                related = await self.search(content, type=MemoryType.KNOWLEDGE, top_k=3)
                if not related:
                    await self.insert(MemoryType.KNOWLEDGE, content)
                else:
                    integrated = await self._integrate_memories(
                        [r["detail"] for r in related],
                        content,
                        MemoryType.KNOWLEDGE,
                        provider, model
                    )
                    # Soft delete old and insert new
                    for r in related:
                        await self.delete(r["id"])
                    await self.insert(MemoryType.KNOWLEDGE, integrated, read_times=10)

            # 2.3 DECISIONS - search then INSERT or UPDATE
            for decision in extracted.get("decisions", []):
                if isinstance(decision, dict):
                    content = self._format_decision(decision)
                else:
                    content = str(decision)

                related = await self.search(content, type=MemoryType.DECISIONS, top_k=3)
                if not related:
                    await self.insert(MemoryType.DECISIONS, content)
                else:
                    integrated = await self._integrate_memories(
                        [r["detail"] for r in related],
                        content,
                        MemoryType.DECISIONS,
                        provider, model
                    )
                    for r in related:
                        await self.delete(r["id"])
                    await self.insert(MemoryType.DECISIONS, integrated, read_times=10)

            # 2.4 PROJECTS - search then INSERT or UPDATE
            for project in extracted.get("projects", []):
                if isinstance(project, dict):
                    content = self._format_project(project)
                else:
                    content = str(project)

                related = await self.search(content, type=MemoryType.PROJECTS, top_k=3)
                if not related:
                    await self.insert(MemoryType.PROJECTS, content)
                else:
                    integrated = await self._integrate_memories(
                        [r["detail"] for r in related],
                        content,
                        MemoryType.PROJECTS,
                        provider, model
                    )
                    for r in related:
                        await self.delete(r["id"])
                    await self.insert(MemoryType.PROJECTS, integrated, read_times=10)

            # 2.5 PROFILE - LLM 整合后覆盖写文件
            if extracted.get("profile_updates"):
                profile_updates = extracted["profile_updates"]
                if isinstance(profile_updates, list):
                    updates_text = "\n".join(
                        u.get("content", str(u)) if isinstance(u, dict) else str(u)
                        for u in profile_updates
                    )
                else:
                    updates_text = str(profile_updates)

                # 读取原 PROFILE.md
                profile_path = self._workspace / "memory" / "PROFILE.md"
                original_profile = ""
                if profile_path.exists():
                    original_profile = profile_path.read_text(encoding="utf-8")

                # LLM 整合
                new_profile = await self._integrate_profile(original_profile, updates_text, provider, model)

                # 覆盖写
                profile_path.parent.mkdir(parents=True, exist_ok=True)
                profile_path.write_text(new_profile, encoding="utf-8")
                logger.info("[MemoryStore] PROFILE.md updated")

            process_elapsed = (time.perf_counter() - process_start) * 1000
            logger.info("[MemoryStore] Consolidation complete: process={}ms", process_elapsed)

            # Update session to mark messages as consolidated
            session.last_consolidated = len(session.messages)

            return True

        except Exception as e:
            logger.error("[MemoryStore] Consolidation failed: {}", e)
            return False

    async def _extract_memories(self, conversation: str, provider: "LLMProvider", model: str) -> dict:
        """
        Extract memories from conversation using LLM.

        Args:
            conversation: Conversation text
            provider: LLM provider
            model: Model name

        Returns:
            Extracted memories dict
        """
        prompt = f"""你是一个记忆提取系统。从以下对话中提取结构化记忆。

## 记忆类型

1. **HISTORY** - 事件日志：简单的时间线记录
2. **KNOWLEDGE** - 知识/经验：可包含来源
3. **DECISIONS** - 决策记录：场景、决定、结果
4. **PROJECTS** - 项目知识：项目信息、任务进度
5. **PROFILE** - 用户画像更新：用户偏好、研究方向等变化

## 格式要求

请以JSON格式返回，包含以下字段：
- history: 字符串，历史事件摘要（如果有）
- knowledge: 数组，知识条目列表（每个包含content字段）
- decisions: 数组，决策列表（每个包含scenario, decision, result字段）
- projects: 数组，项目列表（每个包含name, progress, tasks字段）
- profile_updates: 数组，用户画像更新内容列表（每个包含content字段）

## 重要：简洁要求

每个提取的记忆条目都要简洁，每个 content/摘要 控制在 300 字符以内，突出核心要点，不要冗余。

## 对话内容

{conversation}

## 输出

请直接返回JSON，不要包含其他文字。
"""

        try:
            response = await provider.chat(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0.3,
            )

            content = response.content if hasattr(response, "content") else str(response)

            # Parse JSON from response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                import json
                return json.loads(json_match.group())

            return {}
        except Exception as e:
            logger.warning("[MemoryStore] Memory extraction failed: {}", e)
            return {}

    async def _integrate_memories(
        self,
        old_memories: list[str],
        new_memory: str,
        memory_type: MemoryType,
        provider: "LLMProvider",
        model: str,
    ) -> str:
        """
        Integrate old memories with new memory using LLM.

        Args:
            old_memories: List of existing memory contents
            new_memory: New memory content to integrate
            memory_type: Type of memory
            provider: LLM provider
            model: Model name

        Returns:
            Integrated memory content
        """
        prompt = f"""你是一个记忆整合系统。将新的记忆与已有的相关记忆整合。

## 记忆类型
{memory_type.value}

## 已有关联记忆
{chr(10).join(f'- {m}' for m in old_memories)}

## 新记忆
{new_memory}

## 任务
整合以上记忆，生成一个新的、更完整的记忆。保留所有重要信息，去除重复。

请直接返回整合后的记忆内容，不要包含解释。
"""

        try:
            response = await provider.chat(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0.3,
            )

            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.warning("[MemoryStore] Memory integration failed: {}", e)
            return new_memory

    def _format_decision(self, decision: dict) -> str:
        """Format decision dict as Markdown."""
        scenario = decision.get("scenario", "")
        dec = decision.get("decision", "")
        result = decision.get("result", "")
        lesson = decision.get("lesson", "")

        content = f"## 场景\n{scenario}\n\n## 决定\n{dec}\n\n## 结果\n{result}"
        if lesson:
            content += f"\n\n## 教训\n{lesson}"
        return content

    def _format_project(self, project: dict) -> str:
        """Format project dict as Markdown."""
        name = project.get("name", "")
        overview = project.get("overview", "")
        progress = project.get("progress", "")
        tasks = project.get("tasks", [])

        content = f"## {name}\n\n### 概述\n{overview}\n\n### 当前阶段\n{progress}\n\n### 任务进度"
        if tasks:
            for task in tasks:
                if isinstance(task, dict):
                    done = "x" if task.get("done") else " "
                    content += f"\n- [{done}] {task.get('name', '')}"
                else:
                    content += f"\n- [ ] {task}"
        return content

    async def _integrate_profile(
        self,
        original: str,
        updates: str,
        provider: "LLMProvider",
        model: str,
    ) -> str:
        """
        Integrate profile updates with original profile.

        Args:
            original: Original PROFILE.md content
            updates: New profile updates text
            provider: LLM provider
            model: Model name

        Returns:
            Integrated profile content
        """
        prompt = f"""你是一个用户画像整合系统。将新的用户画像更新与原始画像整合。

## 原始用户画像
{original if original else "(空)"}

## 新的更新内容
{updates}

## 任务
1. 理解原始用户画像的结构
2. 将新的更新内容融入其中
3. 生成更新后的完整用户画像
4. 保留原有的合理内容
5. 如果更新内容与原内容冲突，以新的为准

请直接返回整合后的用户画像内容（Markdown 格式），不要包含解释。
"""

        try:
            response = await provider.chat(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0.3,
            )
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.warning("[MemoryStore] Profile integration failed: {}", e)
            # Fallback: append updates to original
            return f"{original}\n\n---\n\n## 更新\n{updates}"

    def _build_conversation_text(self, messages: list[dict]) -> str:
        """Build conversation text from messages."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content:
                parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def get_memory_context(self) -> str:
        """
        Get PROFILE.md content for prompt.

        Returns:
            PROFILE.md file content, or empty string if not found.
        """
        profile_path = self._workspace / "memory" / "PROFILE.md"
        if profile_path.exists():
            return profile_path.read_text(encoding="utf-8")
        return ""

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Sanitize query for FTS5 simple phrase search."""
        # Replace anything that's not a letter, number, or Chinese with space
        sanitized = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff]', ' ', query)
        # Normalize multiple spaces to single space
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()

        if not sanitized:
            return ""

        # Return as a single quoted phrase
        return f'"{sanitized}"'

    @property
    def connection(self):
        """Public access to database connection."""
        return self._db.connection

    def close(self) -> None:
        """Close database connections."""
        self._db.close()
        logger.debug("[MemoryStore] Closed")

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
