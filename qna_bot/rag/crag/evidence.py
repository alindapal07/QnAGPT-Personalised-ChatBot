"""Evidence aggregation, cross-source synthesis, and citation bundling for CRAG."""

from typing import Any, Dict, List, Optional, Sequence

from qna_bot.rag.crag.models import (
    CRAGDocument,
    CRAGEvaluation,
    EvidenceBundle,
    EvidenceItem,
    EvidenceStatus,
    RelevanceGrade,
)


class EvidenceAssembler:
    """Assembles, formats, and validates evidence bundles for answer generation."""

    def __init__(
        self,
        relevance_threshold: float = 0.70,
        confidence_threshold: float = 0.70,
        coverage_threshold: float = 0.65,
    ) -> None:
        self.relevance_threshold = relevance_threshold
        self.confidence_threshold = confidence_threshold
        self.coverage_threshold = coverage_threshold

    def assemble_from_evaluation(
        self,
        evaluation: CRAGEvaluation,
        documents: Sequence[CRAGDocument],
        query: str = "",
    ) -> EvidenceBundle:
        """Construct an EvidenceBundle from document evaluation output."""
        if not documents or evaluation.relevance_grade == RelevanceGrade.INCORRECT:
            return EvidenceBundle(
                status=EvidenceStatus.EMPTY if not documents else EvidenceStatus.INSUFFICIENT,
                items=[],
                synthesized_context="",
                overall_confidence=evaluation.confidence_score,
                query_coverage=evaluation.query_coverage,
                sources=[],
                contradiction_detected=evaluation.contradiction_detected,
                contradiction_notes=evaluation.contradiction_details,
                missing_aspects=evaluation.missing_aspects,
                requires_web_search=True,
                requires_rewrite=False,
            )

        # Retain only relevant documents
        relevant_docs = [
            documents[i] for i in evaluation.relevant_indices
            if 0 <= i < len(documents)
        ]
        if not relevant_docs and evaluation.relevance_grade != RelevanceGrade.INCORRECT:
            # Fallback to documents with score >= threshold
            relevant_docs = [d for d in documents if d.score >= 0.35]

        items: List[EvidenceItem] = []
        sources: List[Dict[str, Any]] = []
        seen_sources = set()

        for doc in relevant_docs:
            item = EvidenceItem(
                text=doc.content,
                source=doc.source,
                document_id=doc.document_id,
                page=doc.page,
                score=doc.score,
                is_web=False,
                metadata=doc.metadata,
            )
            items.append(item)

            src_key = (doc.source, doc.page)
            if src_key not in seen_sources:
                seen_sources.add(src_key)
                src_dict: Dict[str, Any] = {
                    "source": doc.source,
                    "document_id": doc.document_id,
                    "score": round(doc.score, 3),
                }
                if doc.page is not None:
                    src_dict["page"] = doc.page
                sources.append(src_dict)

        # Determine bundle status
        if evaluation.contradiction_detected:
            status = EvidenceStatus.CONTRADICTORY
            requires_web = True
            requires_rewrite = False
        elif (
            evaluation.relevance_grade == RelevanceGrade.CORRECT
            and evaluation.query_coverage >= self.coverage_threshold
        ):
            status = EvidenceStatus.VALID
            requires_web = False
            requires_rewrite = False
        elif evaluation.relevance_grade == RelevanceGrade.AMBIGUOUS:
            status = EvidenceStatus.PARTIAL
            requires_web = False
            requires_rewrite = True
        else:
            status = EvidenceStatus.INSUFFICIENT
            requires_web = True
            requires_rewrite = False

        # Build synthesized context string
        context_lines: List[str] = []
        for idx, item in enumerate(items, 1):
            src_label = item.source
            if item.page is not None:
                src_label += f" (Page {item.page})"
            context_lines.append(f"[{idx}] Source: {src_label} (Relevance: {item.score:.2f})\n{item.text.strip()}")

        synthesized_context = "\n\n".join(context_lines)

        return EvidenceBundle(
            status=status,
            items=items,
            synthesized_context=synthesized_context,
            overall_confidence=evaluation.confidence_score,
            query_coverage=evaluation.query_coverage,
            sources=sources,
            contradiction_detected=evaluation.contradiction_detected,
            contradiction_notes=evaluation.contradiction_details,
            missing_aspects=evaluation.missing_aspects,
            requires_web_search=requires_web,
            requires_rewrite=requires_rewrite,
        )

    def assemble_from_web_results(
        self,
        web_results: Sequence[Dict[str, Any]],
        opened_pages: Optional[Sequence[Dict[str, Any]]] = None,
        query: str = "",
    ) -> EvidenceBundle:
        """Construct an EvidenceBundle from web search hits and fetched web pages."""
        items: List[EvidenceItem] = []
        sources: List[Dict[str, Any]] = []
        seen_urls = set()

        # Prioritize full text from opened pages
        pages = opened_pages or []
        for page in pages:
            url = page.get("url", "")
            title = page.get("title", "") or url
            text = page.get("content", "") or page.get("text", "")
            if text and url not in seen_urls:
                seen_urls.add(url)
                items.append(
                    EvidenceItem(
                        text=text[:3500],
                        source=f"{title} ({url})",
                        is_web=True,
                        score=0.85,
                        metadata={"url": url, "title": title},
                    )
                )
                sources.append({"source": title, "url": url, "type": "web"})

        # Supplement with search snippets
        for res in web_results:
            url = res.get("url", "")
            title = res.get("title", "") or url
            snippet = res.get("snippet", "")
            if url and url not in seen_urls and snippet:
                seen_urls.add(url)
                items.append(
                    EvidenceItem(
                        text=snippet,
                        source=f"{title} ({url})",
                        is_web=True,
                        score=0.70,
                        metadata={"url": url, "title": title},
                    )
                )
                sources.append({"source": title, "url": url, "type": "web"})

        if not items:
            return EvidenceBundle(
                status=EvidenceStatus.EMPTY,
                items=[],
                synthesized_context="",
                overall_confidence=0.0,
                query_coverage=0.0,
                sources=[],
                requires_web_search=False,
                requires_rewrite=False,
            )

        context_lines: List[str] = []
        for idx, item in enumerate(items, 1):
            context_lines.append(f"[{idx}] Web Source: {item.source}\n{item.text.strip()}")

        return EvidenceBundle(
            status=EvidenceStatus.VALID,
            items=items,
            synthesized_context="\n\n".join(context_lines),
            overall_confidence=0.80,
            query_coverage=0.85,
            sources=sources,
            requires_web_search=False,
            requires_rewrite=False,
        )
