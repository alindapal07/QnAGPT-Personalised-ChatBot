"""Pydantic v2 data models for Corrective Retrieval-Augmented Generation (CRAG)."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class RelevanceGrade(str, Enum):
    """Relevance classification for retrieved documents or chunks."""

    CORRECT = "CORRECT"
    AMBIGUOUS = "AMBIGUOUS"
    INCORRECT = "INCORRECT"
    CONTRADICTORY = "CONTRADICTORY"


class EvidenceStatus(str, Enum):
    """Aggregate status of the gathered evidence bundle."""

    VALID = "VALID"
    PARTIAL = "PARTIAL"
    INSUFFICIENT = "INSUFFICIENT"
    CONTRADICTORY = "CONTRADICTORY"
    EMPTY = "EMPTY"


class CRAGDecision(str, Enum):
    """Action decision produced by CRAG router."""

    CORRECT = "CORRECT"
    AMBIGUOUS = "AMBIGUOUS"
    INCORRECT = "INCORRECT"
    CONTRADICTORY = "CONTRADICTORY"


def _clamp_unit_interval(v: float) -> float:
    """Safely clamp float values to [0.0, 1.0]."""
    try:
        val = float(v)
    except (ValueError, TypeError):
        return 0.0
    return max(0.0, min(1.0, val))


class CRAGDocument(BaseModel):
    """Normalized document chunk representation within the CRAG pipeline."""

    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    document_id: str = ""
    source: str = "Unknown"
    page: Optional[int] = None
    chunk_index: Optional[int] = None

    @field_validator("score", mode="before")
    @classmethod
    def clamp_score(cls, v: Any) -> float:
        return _clamp_unit_interval(v)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
            "document_id": self.document_id,
            "source": self.source,
            "page": self.page,
            "chunk_index": self.chunk_index,
        }


class CRAGEvaluation(BaseModel):
    """Multi-dimensional evaluation of retrieved documents."""

    relevance_grade: RelevanceGrade
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    query_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    factual_sufficiency: float = Field(default=0.0, ge=0.0, le=1.0)
    contradiction_detected: bool = False
    temporal_validity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    relevant_indices: List[int] = Field(default_factory=list)
    salient_snippets: List[str] = Field(default_factory=list)
    missing_aspects: List[str] = Field(default_factory=list)
    contradiction_details: List[str] = Field(default_factory=list)
    explanation: str = ""
    used_llm: bool = True

    @field_validator(
        "relevance_score",
        "confidence_score",
        "query_coverage",
        "factual_sufficiency",
        mode="before",
    )
    @classmethod
    def clamp_required_scores(cls, v: Any) -> float:
        return _clamp_unit_interval(v)

    @field_validator("temporal_validity", mode="before")
    @classmethod
    def clamp_optional_scores(cls, v: Any) -> Optional[float]:
        if v is None:
            return None
        return _clamp_unit_interval(v)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relevance_grade": self.relevance_grade.value,
            "relevance_score": self.relevance_score,
            "confidence_score": self.confidence_score,
            "query_coverage": self.query_coverage,
            "factual_sufficiency": self.factual_sufficiency,
            "contradiction_detected": self.contradiction_detected,
            "temporal_validity": self.temporal_validity,
            "relevant_indices": self.relevant_indices,
            "salient_snippets": self.salient_snippets,
            "missing_aspects": self.missing_aspects,
            "contradiction_details": self.contradiction_details,
            "explanation": self.explanation,
            "used_llm": self.used_llm,
        }


class RewrittenQuery(BaseModel):
    """Structured output of the query rewriter."""

    original_query: str
    rewritten_query: str
    subqueries: List[str] = Field(default_factory=list)
    extracted_keywords: List[str] = Field(default_factory=list)
    rationale: str = ""
    attempt: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "rewritten_query": self.rewritten_query,
            "subqueries": self.subqueries,
            "extracted_keywords": self.extracted_keywords,
            "rationale": self.rationale,
            "attempt": self.attempt,
        }


class EvidenceItem(BaseModel):
    """A granular piece of evidence extracted from a document or webpage."""

    text: str
    source: str
    document_id: Optional[str] = None
    page: Optional[int] = None
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    is_web: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("score", mode="before")
    @classmethod
    def clamp_item_score(cls, v: Any) -> float:
        return _clamp_unit_interval(v)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "document_id": self.document_id,
            "page": self.page,
            "score": self.score,
            "is_web": self.is_web,
            "metadata": self.metadata,
        }


class EvidenceBundle(BaseModel):
    """Aggregated evidence payload passed to answer generation."""

    status: EvidenceStatus = EvidenceStatus.EMPTY
    items: List[EvidenceItem] = Field(default_factory=list)
    synthesized_context: str = ""
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    query_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    contradiction_detected: bool = False
    contradiction_notes: List[str] = Field(default_factory=list)
    missing_aspects: List[str] = Field(default_factory=list)
    requires_web_search: bool = False
    requires_rewrite: bool = False

    @field_validator("overall_confidence", "query_coverage", mode="before")
    @classmethod
    def clamp_bundle_scores(cls, v: Any) -> float:
        return _clamp_unit_interval(v)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "items": [item.to_dict() for item in self.items],
            "synthesized_context": self.synthesized_context,
            "overall_confidence": self.overall_confidence,
            "query_coverage": self.query_coverage,
            "sources": self.sources,
            "contradiction_detected": self.contradiction_detected,
            "contradiction_notes": self.contradiction_notes,
            "missing_aspects": self.missing_aspects,
            "requires_web_search": self.requires_web_search,
            "requires_rewrite": self.requires_rewrite,
        }
