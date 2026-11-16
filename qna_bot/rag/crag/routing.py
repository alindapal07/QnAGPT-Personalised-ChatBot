"""Deterministic decision router for Corrective RAG (CRAG)."""

from typing import Literal

from qna_bot.rag.crag.models import CRAGDecision, CRAGEvaluation, RelevanceGrade


class CRAGRouter:
    """Evaluates CRAG state and decides next workflow destination with bounded loops."""

    def __init__(
        self,
        max_retrieval_attempts: int = 2,
        max_web_search_attempts: int = 1,
        web_fallback_enabled: bool = True,
    ) -> None:
        self.max_retrieval_attempts = max_retrieval_attempts
        self.max_web_search_attempts = max_web_search_attempts
        self.web_fallback_enabled = web_fallback_enabled

    def decide(
        self,
        evaluation: CRAGEvaluation,
        retrieval_attempts: int = 1,
        web_search_attempts: int = 0,
        chat_mode: str = "Auto (Agentic)",
    ) -> CRAGDecision:
        """Route to CORRECT, AMBIGUOUS, INCORRECT, or CONTRADICTORY."""
        # 1. Contradiction detected
        if evaluation.contradiction_detected:
            if (
                self.web_fallback_enabled
                and web_search_attempts < self.max_web_search_attempts
                and chat_mode == "Auto (Agentic)"
            ):
                return CRAGDecision.CONTRADICTORY
            return CRAGDecision.CORRECT  # Synthesize grounded answer explicitly noting the contradiction

        # 2. Correct relevance
        if evaluation.relevance_grade == RelevanceGrade.CORRECT:
            return CRAGDecision.CORRECT

        # 3. Ambiguous relevance
        if evaluation.relevance_grade == RelevanceGrade.AMBIGUOUS:
            if retrieval_attempts < self.max_retrieval_attempts:
                return CRAGDecision.AMBIGUOUS
            # Retrieval attempts exhausted: fall back to web search if allowed, else synthesize
            if (
                self.web_fallback_enabled
                and web_search_attempts < self.max_web_search_attempts
                and chat_mode == "Auto (Agentic)"
            ):
                return CRAGDecision.INCORRECT
            return CRAGDecision.CORRECT

        # 4. Incorrect relevance
        if (
            self.web_fallback_enabled
            and web_search_attempts < self.max_web_search_attempts
            and chat_mode == "Auto (Agentic)"
        ):
            return CRAGDecision.INCORRECT

        # Fallback to answer generation (which will abstain if evidence is insufficient)
        return CRAGDecision.CORRECT
