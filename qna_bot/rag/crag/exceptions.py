"""Typed exceptions for the Corrective Retrieval-Augmented Generation (CRAG) pipeline."""


class CRAGError(Exception):
    """Base exception for all CRAG-related operational failures."""


class EvaluationError(CRAGError):
    """Raised when document evaluation fails or returns malformed results."""


class QueryRewriteError(CRAGError):
    """Raised when query rewriting fails or produces unusable search strings."""


class EvidenceThresholdError(CRAGError):
    """Raised when gathered evidence fails to satisfy required confidence/coverage thresholds."""


class WebFallbackError(CRAGError):
    """Raised when web search fallback fails to retrieve viable content."""
