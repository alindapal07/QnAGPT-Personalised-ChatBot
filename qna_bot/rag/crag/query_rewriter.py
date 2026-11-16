"""Context-aware, drift-resistant Query Rewriter and Decomposer for CRAG."""

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from qna_bot.config.settings import AppSettings, get_settings
from qna_bot.core.logging_config import get_logger
from qna_bot.rag.crag.models import CRAGEvaluation, RewrittenQuery

logger = get_logger(__name__)

_FILLER_PATTERNS = [
    r"^(?:please\s+)?(?:can|could|would)\s+you\s+(?:please\s+)?(?:tell|explain|show|give|find|describe)\s+(?:me\s+)?",
    r"^(?:i\s+(?:want|need|would\s+like)\s+to\s+know\s+)",
    r"^(?:what\s+(?:is|are|were|was)\s+)",
    r"^(?:tell\s+me\s+about\s+)",
    r"^(?:show\s+me\s+)",
    r"\?+$",
]


class _LLMRewriteSchema(BaseModel):
    rewritten_query: str = Field(description="Clear, standalone semantic search query.")
    subqueries: List[str] = Field(default_factory=list, description="Decomposed subqueries if multi-faceted.")
    extracted_keywords: List[str] = Field(default_factory=list, description="Key domain concepts and entities.")
    rationale: str = Field(default="", description="Brief justification for rewriting.")


def _clean_history_text(messages: Sequence[BaseMessage], limit: int = 4) -> str:
    """Format recent conversational turns into clean role: content strings."""
    relevant = [
        m for m in messages
        if isinstance(m, (HumanMessage, AIMessage)) and str(getattr(m, "content", "")).strip()
    ][-limit:]
    lines = []
    for m in relevant:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        snippet = str(m.content).strip().replace("\n", " ")[:200]
        lines.append(f"{role}: {snippet}")
    return "\n".join(lines)


class QueryRewriter:
    """Reformulates ambiguous or multi-faceted queries into targeted retrieval queries."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        settings: Optional[AppSettings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.model_name = model_name or self.settings.default_text_model
        self.api_key = (api_key if api_key is not None else self.settings.groq_api_key).strip()

    def _make_llm(self) -> ChatGroq:
        return ChatGroq(
            model=self.model_name,
            temperature=0.0,
            max_tokens=384,
            api_key=self.api_key,
        )

    def rewrite(
        self,
        query: str,
        messages: Optional[Sequence[BaseMessage]] = None,
        previous_rewrites: Optional[Sequence[str]] = None,
        attempt: int = 1,
        evaluation: Optional[CRAGEvaluation] = None,
    ) -> RewrittenQuery:
        """Formulate a standalone semantic query with subqueries and keywords."""
        cleaned = (query or "").strip()
        if not cleaned:
            return RewrittenQuery(
                original_query="",
                rewritten_query="",
                subqueries=[],
                extracted_keywords=[],
                rationale="Empty query provided",
                attempt=attempt,
            )

        if not self.api_key:
            return self._heuristic_rewrite(cleaned, attempt, evaluation)

        try:
            return self._llm_rewrite(cleaned, messages or [], previous_rewrites or [], attempt, evaluation)
        except Exception as exc:
            logger.warning("LLM query rewrite failed (%s); using heuristic fallback.", exc)
            return self._heuristic_rewrite(cleaned, attempt, evaluation)

    def _llm_rewrite(
        self,
        query: str,
        messages: Sequence[BaseMessage],
        previous_rewrites: Sequence[str],
        attempt: int,
        evaluation: Optional[CRAGEvaluation],
    ) -> RewrittenQuery:
        history_str = _clean_history_text(messages)
        prev_str = "; ".join(previous_rewrites) if previous_rewrites else "None"

        guidance = [
            "You are an expert retrieval query optimizer for Corrective RAG (CRAG).",
            "Rewrite the user query into a concise, standalone semantic retrieval query.",
            "Rules:",
            "1. Resolve pronouns (it, they, that, their) using the conversation context.",
            "2. Preserve all core domain nouns, numbers, names, and technical terms.",
            "3. Do NOT hallucinate new company names or topics not mentioned by the user.",
            "4. If the query asks multiple things, split into distinct subqueries.",
        ]

        if attempt > 1:
            guidance.append(
                f"Attempt #{attempt}: Previous searches ({prev_str}) yielded insufficient or ambiguous results."
            )
            if evaluation and evaluation.missing_aspects:
                aspects_str = ", ".join(evaluation.missing_aspects)
                guidance.append(f"Explicitly target these missing aspects: {aspects_str}")
            guidance.append("Broaden search terms and include essential domain synonyms.")

        prompt = (
            f"{' '.join(guidance)}\n\n"
            f"Conversation History:\n{history_str or 'None'}\n\n"
            f"User Query:\n{query}\n\n"
            "Return JSON matching the schema."
        )

        llm = self._make_llm()
        parsed: Optional[_LLMRewriteSchema] = None
        try:
            structured = llm.with_structured_output(_LLMRewriteSchema)
            res = structured.invoke(prompt)
            if isinstance(res, _LLMRewriteSchema):
                parsed = res
            elif isinstance(res, dict):
                parsed = _LLMRewriteSchema.model_validate(res)
        except Exception as exc:
            logger.info("Structured query rewrite invoke failed: %s", exc)
            raw = llm.invoke(prompt)
            match = re.search(r"\{.*\}", str(getattr(raw, "content", "")), flags=re.DOTALL)
            if match:
                try:
                    parsed = _LLMRewriteSchema.model_validate(json.loads(match.group(0)))
                except Exception:
                    pass

        if not parsed or not parsed.rewritten_query.strip():
            return self._heuristic_rewrite(query, attempt, evaluation)

        return RewrittenQuery(
            original_query=query,
            rewritten_query=parsed.rewritten_query.strip(),
            subqueries=[sq.strip() for sq in parsed.subqueries if sq.strip()],
            extracted_keywords=[kw.strip() for kw in parsed.extracted_keywords if kw.strip()],
            rationale=parsed.rationale or "Optimized by LLM rewriter.",
            attempt=attempt,
        )

    def _heuristic_rewrite(
        self,
        query: str,
        attempt: int,
        evaluation: Optional[CRAGEvaluation],
    ) -> RewrittenQuery:
        """Deterministic keyword-preserving query optimizer."""
        cleaned = query.strip()
        for pat in _FILLER_PATTERNS:
            cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()

        # Extract keywords
        raw_words = re.findall(r"[a-zA-Z0-9_-]{2,}", cleaned)
        keywords = [w.lower() for w in raw_words if len(w) > 2]

        # Decompose multi-part queries
        subqueries: List[str] = []
        parts = re.split(r"\s+(?:and|or|as well as|compared to|versus|vs\.?)\s+", cleaned, flags=re.IGNORECASE)
        if len(parts) > 1:
            subqueries = [p.strip() for p in parts if len(p.strip()) > 5]

        # If retry, append missing aspects
        rewritten = cleaned or query
        if attempt > 1 and evaluation and evaluation.missing_aspects:
            extra = " ".join(evaluation.missing_aspects[:2])
            rewritten = f"{rewritten} {extra}".strip()

        return RewrittenQuery(
            original_query=query,
            rewritten_query=rewritten,
            subqueries=subqueries,
            extracted_keywords=keywords[:8],
            rationale=f"Heuristically refined (attempt {attempt}).",
            attempt=attempt,
        )
