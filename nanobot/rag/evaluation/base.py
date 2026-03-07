"""RAG Evaluation - Base data structures."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EvalQuery:
    """Single test query for RAG evaluation."""
    id: str
    query: str
    golden_context: str
    source_chunk_id: Optional[int] = None
    source_doc: Optional[str] = None
    difficulty: Optional[str] = None  # easy/medium/hard
    tags: Optional[List[str]] = None
    golden_embedding: Optional[List[float]] = None


@dataclass
class EvalResult:
    """Single query evaluation result."""
    query_id: str
    query: str
    hit: bool
    hit_rank: Optional[int] = None
    hit_reason: Optional[str] = None  # "id_match" / "strong_semantic" / "weak_semantic"
    failure_reason: Optional[str] = None
    similarity_scores: Optional[List[float]] = None
    best_similarity: Optional[float] = None  # Best similarity score among results
    found_chunk_ids: Optional[List[int]] = None
    latency_ms: float = 0.0
    difficulty: Optional[str] = None
    # Baseline comparison
    baseline_hit: Optional[bool] = None
    baseline_hit_rank: Optional[int] = None


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    strong_threshold: float = 0.7
    weak_threshold: float = 0.6
    top_k: int = 5
    random_seed: Optional[int] = 42


@dataclass
class EvalSummary:
    """Evaluation summary."""
    dataset_name: str
    num_queries: int
    config: EvalConfig
    # Core metrics
    recall_at_5: float
    mrr: float
    hit_rate_at_5: float
    avg_latency_ms: float
    random_seed: Optional[int] = None
    # Baseline comparison metrics
    baseline_recall_at_5: Optional[float] = None
    baseline_mrr: Optional[float] = None
    # Extended metrics
    difficulty_breakdown: Optional[dict] = None
    failure_breakdown: Optional[dict] = None
    # Detailed results
    details: Optional[List[EvalResult]] = None
