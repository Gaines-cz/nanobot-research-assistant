"""RAG Evaluation module for nanobot."""

from nanobot.rag.evaluation.ablation import AblationConfig, ABLATION_CONFIGS
from nanobot.rag.evaluation.base import (
    EvalConfig,
    EvalQuery,
    EvalResult,
    EvalSummary,
)
from nanobot.rag.evaluation.baseline import BaselineRetriever
from nanobot.rag.evaluation.evaluator import AblationStudy, RAGEvaluator
from nanobot.rag.evaluation.generator import DataGenerator
from nanobot.rag.evaluation.judge import ResultJudge
from nanobot.rag.evaluation.metrics import MetricsCalculator
from nanobot.rag.evaluation.report import (
    FailureAnalysis,
    FailureAnalyzer,
    FailureCategory,
    ReportGenerator,
)

__all__ = [
    "EvalQuery",
    "EvalResult",
    "EvalSummary",
    "EvalConfig",
    "DataGenerator",
    "ResultJudge",
    "MetricsCalculator",
    "BaselineRetriever",
    "RAGEvaluator",
    "AblationStudy",
    "AblationConfig",
    "ABLATION_CONFIGS",
    "ReportGenerator",
    "FailureAnalyzer",
    "FailureAnalysis",
    "FailureCategory",
]
