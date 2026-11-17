"""Corrective RAG (CRAG) evaluation and document grading node."""

from typing import Any, Dict

from qna_bot.agent.nodes.base import (
    BaseNodeSuite,
    _deduplicate_doc_sources,
    _format_document_context,
)
from qna_bot.agent.state import AgentState
from qna_bot.rag.crag import (
    CRAGDecision,
    CRAGDocument,
)
from qna_bot.rag.vector_store import RetrievedChunk


class CRAGNodesMixin:
    """Mixin containing document grading and CRAG evaluation node."""

    def grade_documents(self: BaseNodeSuite, state: AgentState) -> Dict[str, Any]:
        """Filter retrieved chunks for semantic relevance, evaluate via CRAG, and build citations."""
        query = (state.get("rewritten_query") or state.get("user_query") or "").strip()
        raw_docs = state.get("retrieved_documents") or []

        if getattr(self.settings, "crag_enabled", True):
            crag_docs = [
                CRAGDocument(
                    content=str(d.get("content", "")),
                    metadata=dict(d.get("metadata") or {}),
                    score=float(d.get("score", 0.0)),
                    document_id=str(d.get("document_id", "")),
                    source=str(d.get("source", "Unknown")),
                    page=d.get("page"),
                    chunk_index=d.get("chunk_index"),
                )
                for d in raw_docs
            ]
            chat_mode = state.get("chat_mode", "Auto (Agentic)")
            retrieval_attempts = int(state.get("retrieval_attempt_count", 1))
            web_attempts = int(state.get("web_retry_count", 0))

            evaluation, bundle, decision = self.crag_pipeline.evaluate_and_assemble(
                query=query,
                documents=crag_docs,
                retrieval_attempts=retrieval_attempts,
                web_search_attempts=web_attempts,
                chat_mode=chat_mode,
            )

            relevant_dicts = [
                {
                    "content": item.text,
                    "metadata": item.metadata,
                    "score": item.score,
                    "document_id": item.document_id or "",
                    "source": item.source,
                    "page": item.page,
                }
                for item in bundle.items
                if not item.is_web
            ]

            citations = bundle.sources
            context_str = bundle.synthesized_context
            retry_count = int(state.get("retry_count", 0))
            if decision == CRAGDecision.AMBIGUOUS:
                retry_count += 1

            return {
                "relevant_documents": relevant_dicts,
                "sources": citations,
                "context": context_str,
                "retry_count": retry_count,
                "document_evaluations": [evaluation.to_dict()],
                "evidence_bundle": bundle.to_dict(),
                "crag_status": bundle.status.value,
                "crag_decision": decision.value,
                "contradiction_detected": bundle.contradiction_detected,
                "activity": f"CRAG evaluated documents as {evaluation.relevance_grade.value}...",
            }

        chunks = [
            RetrievedChunk(
                content=str(d.get("content", "")),
                metadata=dict(d.get("metadata") or {}),
                score=float(d.get("score", 0.0)),
                document_id=str(d.get("document_id", "")),
                source=str(d.get("source", "Unknown")),
                page=d.get("page"),
            )
            for d in raw_docs
        ]

        grading = self.grader.grade(query=query, chunks=chunks)
        relevant_dicts = [
            {
                "content": c.content,
                "metadata": c.metadata,
                "score": c.score,
                "document_id": c.document_id,
                "source": c.source,
                "page": c.page,
            }
            for c in grading.relevant_chunks
        ]
        citations = _deduplicate_doc_sources(grading.relevant_chunks)
        context_str = _format_document_context(relevant_dicts)

        retry_count = int(state.get("retry_count", 0))
        if not relevant_dicts:
            retry_count += 1

        return {
            "relevant_documents": relevant_dicts,
            "sources": citations,
            "context": context_str,
            "retry_count": retry_count,
            "activity": "Checking source relevance...",
        }
