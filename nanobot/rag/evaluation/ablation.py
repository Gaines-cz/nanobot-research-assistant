"""RAG Evaluation - Ablation Study Configurations.

This module defines configurations for ablation studies to measure
the contribution of each component in the RAG pipeline.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class AblationConfig:
    """Single ablation configuration for RAG pipeline.

    Each configuration represents a variant of the pipeline with
    certain components enabled or disabled.
    """
    name: str                    # Configuration name, e.g., "BM25 Only"
    description: str             # Description of what this configuration tests

    # Core retrieval components
    enable_bm25: bool = True
    enable_vector: bool = True

    # Query processing
    enable_query_expand: bool = True

    # Post-retrieval processing
    enable_context_expansion: bool = True
    enable_document_level: bool = True
    enable_rerank: bool = True

    def apply_to_rag_config(self, rag_config) -> None:
        """Apply this ablation configuration to a RAGConfig object.

        Args:
            rag_config: RAGConfig object to modify
        """
        rag_config.enable_bm25 = self.enable_bm25
        rag_config.enable_vector = self.enable_vector
        rag_config.enable_context_expansion = self.enable_context_expansion
        rag_config.enable_document_level = self.enable_document_level
        rag_config.enable_rerank = self.enable_rerank
        rag_config.enable_query_expand = self.enable_query_expand


# Predefined ablation configurations in order of increasing complexity
ABLATION_CONFIGS: List[AblationConfig] = [
    AblationConfig(
        name="BM25 Only (Baseline)",
        description="仅 BM25 全文检索，无其他优化",
        enable_vector=False,
        enable_context_expansion=False,
        enable_document_level=False,
        enable_rerank=False,
        enable_query_expand=False,
    ),
    AblationConfig(
        name="Vector Only",
        description="仅向量检索，无其他优化",
        enable_bm25=False,
        enable_context_expansion=False,
        enable_document_level=False,
        enable_rerank=False,
        enable_query_expand=False,
    ),
    AblationConfig(
        name="Hybrid (BM25+Vector)",
        description="混合检索（BM25 + 向量），无后续优化",
        enable_context_expansion=False,
        enable_document_level=False,
        enable_rerank=False,
        enable_query_expand=False,
    ),
    AblationConfig(
        name="+ Query Expansion",
        description="混合检索 + 查询扩展",
        enable_context_expansion=False,
        enable_document_level=False,
        enable_rerank=False,
    ),
    AblationConfig(
        name="+ Context Expansion",
        description="加上下文扩展",
        enable_document_level=False,
        enable_rerank=False,
    ),
    AblationConfig(
        name="+ Document-Level",
        description="加文档级优先",
        enable_rerank=False,
    ),
    AblationConfig(
        name="Full Pipeline",
        description="完整四阶段流水线",
    ),
]
