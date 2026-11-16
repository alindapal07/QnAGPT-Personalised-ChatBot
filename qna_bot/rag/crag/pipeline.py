"""High-level orchestrator and entry point for Corrective RAG (CRAG)."""

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.messages import BaseMessage

from qna_bot.config.settings import AppSettings, get_settings
from qna_bot.rag.crag.document_filter import DocumentFilter
from qna_bot.rag.crag.evaluator import DocumentEvaluator
from qna_bot.rag.crag.evidence import EvidenceAssembler
from qna_bot.rag.crag.metrics import CRAGMetricsTracker, get_crag_metrics
from qna_bot.rag.crag.models import (
    CRAGDecision,
    CRAGDocument,
    CRAGEvaluation,
    EvidenceBundle,
    RewrittenQuery,
)
from qna_bot.rag.crag.query_rewriter import QueryRewriter
from qna_bot.rag.crag.routing import CRAGRouter


class CRAGPipeline:
    """Production-grade Corrective RAG Pipeline combining evaluation, rewriting, filtering, and evidence bundling."""

    def __init__(
        self,
        evaluator: Optional[DocumentEvaluator] = None,
        query_rewriter: Optional[QueryRewriter] = None,
        document_filter: Optional[DocumentFilter] = None,
        evidence_assembler: Optional[EvidenceAssembler] = None,
        router: Optional[CRAGRouter] = None,
        metrics: Optional[CRAGMetricsTracker] = None,
        settings: Optional[AppSettings] = None,
    ) -> None:
        cfg = settings or get_settings()
        self.settings = cfg
        self.evaluator = evaluator or DocumentEvaluator(
            relevance_threshold=getattr(cfg, "crag_relevance_threshold", 0.70),
            confidence_threshold=getattr(cfg, "crag_confidence_threshold", 0.70),
            coverage_threshold=getattr(cfg, "crag_coverage_threshold", 0.65),
            settings=cfg,
        )
        self.query_rewriter = query_rewriter or QueryRewriter(settings=cfg)
        self.document_filter = document_filter or DocumentFilter()
        self.evidence_assembler = evidence_assembler or EvidenceAssembler(
            relevance_threshold=getattr(cfg, "crag_relevance_threshold", 0.70),
            confidence_threshold=getattr(cfg, "crag_confidence_threshold", 0.70),
            coverage_threshold=getattr(cfg, "crag_coverage_threshold", 0.65),
        )
        self.router = router or CRAGRouter(
            max_retrieval_attempts=getattr(cfg, "crag_max_retrieval_attempts", 2),
            max_web_search_attempts=getattr(cfg, "crag_max_web_search_attempts", 1),
            web_fallback_enabled=getattr(cfg, "crag_web_fallback_enabled", True),
        )
        self.metrics = metrics or get_crag_metrics()

    def evaluate_and_assemble(
        self,
        query: str,
        documents: Sequence[CRAGDocument],
        retrieval_attempts: int = 1,
        web_search_attempts: int = 0,
        chat_mode: str = "Auto (Agentic)",
    ) -> Tuple[CRAGEvaluation, EvidenceBundle, CRAGDecision]:
        """Evaluate candidate documents, build evidence bundle, and determine next route."""
        start_t = time.perf_counter()
        evaluation = self.evaluator.evaluate_batch(query=query, documents=documents)
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        self.metrics.record_evaluation(evaluation, latency_ms=elapsed_ms)

        bundle = self.evidence_assembler.assemble_from_evaluation(
            evaluation=evaluation,
            documents=documents,
            query=query,
        )

        decision = self.router.decide(
            evaluation=evaluation,
            retrieval_attempts=retrieval_attempts,
            web_search_attempts=web_search_attempts,
            chat_mode=chat_mode,
        )

        return evaluation, bundle, decision

    def rewrite_query(
        self,
        query: str,
        messages: Optional[Sequence[BaseMessage]] = None,
        previous_rewrites: Optional[Sequence[str]] = None,
        attempt: int = 1,
        evaluation: Optional[CRAGEvaluation] = None,
    ) -> RewrittenQuery:
        """Formulate an optimized retrieval query and record metric."""
        rewritten = self.query_rewriter.rewrite(
            query=query,
            messages=messages,
            previous_rewrites=previous_rewrites,
            attempt=attempt,
            evaluation=evaluation,
        )
        self.metrics.record_rewrite()
        return rewritten

    def assemble_web_evidence(
        self,
        web_results: Sequence[Dict[str, Any]],
        opened_pages: Optional[Sequence[Dict[str, Any]]] = None,
        query: str = "",
    ) -> EvidenceBundle:
        """Assemble an evidence bundle from web search and extraction results."""
        self.metrics.record_web_fallback()
        return self.evidence_assembler.assemble_from_web_results(
            web_results=web_results,
            opened_pages=opened_pages,
            query=query,
        )
