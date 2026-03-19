"""RAG Evaluation - Base data structures."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EvalQuery:
    """Single test query for RAG evaluation."""
    id: str
    query: str  # LLM 生成的原始问题
    golden_context: str  # 对应的 chunk 内容
    source_chunk_id: Optional[int] = None
    source_doc: Optional[str] = None
    question_type: Optional[str] = None  # factoid / summary / analytical
    tags: List[str] = field(default_factory=list)
    golden_embedding: Optional[List[float]] = None


@dataclass
class EvalResult:
    """Single query evaluation result."""
    query_id: str
    query: str
    hit: bool
    hit_rank: Optional[int] = None
    hit_reason: Optional[str] = None
    failure_reason: Optional[str] = None
    similarity_scores: Optional[List[float]] = None
    best_similarity: Optional[float] = None
    found_chunk_ids: Optional[List[int]] = None
    latency_ms: float = 0.0
    # Baseline comparison
    baseline_hit: Optional[bool] = None
    baseline_hit_rank: Optional[int] = None
    # For NDCG calculation
    relevance_scores: Optional[List[float]] = None  # Relevance of each retrieved chunk (0 or 1)


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    top_k: int = 5
    random_seed: Optional[int] = 42


@dataclass
class EvalSummary:
    """Evaluation summary."""
    dataset_name: str
    num_queries: int
    config: EvalConfig
    # Core metrics
    recall_at_k: float
    mrr: float
    hit_rate_at_k: float
    ndcg_at_k: float
    avg_latency_ms: float
    random_seed: Optional[int] = None
    # Baseline comparison metrics
    baseline_recall_at_k: Optional[float] = None
    baseline_mrr: Optional[float] = None
    baseline_ndcg_at_k: Optional[float] = None
    # Question type breakdown
    question_type_breakdown: Optional[dict] = None
    # Detailed results
    details: Optional[List[EvalResult]] = None


@dataclass
class TestDataset:
    """A complete test dataset."""
    version: str
    created_at: str
    num_queries: int
    queries: List[EvalQuery]
    metadata: dict = field(default_factory=dict)
