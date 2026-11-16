"""Corrective Retrieval-Augmented Generation (CRAG) module."""

from qna_bot.rag.crag.document_filter import DocumentFilter
from qna_bot.rag.crag.evaluator import DocumentEvaluator
from qna_bot.rag.crag.evidence import EvidenceAssembler
from qna_bot.rag.crag.exceptions import (
    CRAGError,
    EvaluationError,
    EvidenceThresholdError,
    QueryRewriteError,
    WebFallbackError,
)
from qna_bot.rag.crag.metrics import CRAGMetricsTracker, get_crag_metrics
from qna_bot.rag.crag.models import (
    CRAGDecision,
    CRAGDocument,
    CRAGEvaluation,
    EvidenceBundle,
    EvidenceItem,
    EvidenceStatus,
    RelevanceGrade,
    RewrittenQuery,
)
from qna_bot.rag.crag.pipeline import CRAGPipeline
from qna_bot.rag.crag.query_rewriter import QueryRewriter
from qna_bot.rag.crag.routing import CRAGRouter

__all__ = [
    "CRAGDecision",
    "CRAGDocument",
    "CRAGEvaluation",
    "CRAGError",
    "CRAGMetricsTracker",
    "CRAGPipeline",
    "CRAGRouter",
    "DocumentEvaluator",
    "DocumentFilter",
    "EvaluationError",
    "EvidenceAssembler",
    "EvidenceBundle",
    "EvidenceItem",
    "EvidenceStatus",
    "EvidenceThresholdError",
    "QueryRewriteError",
    "QueryRewriter",
    "RelevanceGrade",
    "RewrittenQuery",
    "WebFallbackError",
    "get_crag_metrics",
]
