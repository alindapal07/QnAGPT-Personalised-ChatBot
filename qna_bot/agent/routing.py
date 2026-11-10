"""Deterministic conditional routing logic for the LangGraph Agentic RAG workflow."""

from typing import Literal

from qna_bot.agent.state import AgentState


def route_after_analysis(
    state: AgentState,
) -> Literal["direct_answer", "rewrite_query", "web_search"]:
    """Deterministically select the next graph node based on chat mode and classified intent."""
    chat_mode = state.get("chat_mode", "Auto (Agentic)")
    query_type = (state.get("query_type") or "GENERAL").upper()
    doc_required = bool(state.get("document_search_required", False))
    web_required = bool(state.get("web_search_required", False))

    if chat_mode == "General Chat":
        return "direct_answer"

    if chat_mode == "Document Chat":
        return "rewrite_query"

    # Auto (Agentic) mode
    if query_type == "DOCUMENT" or doc_required:
        return "rewrite_query"
    if query_type == "WEB_SEARCH" or web_required:
        return "web_search"
    return "direct_answer"


def route_after_grading(
    state: AgentState,
    max_retrieval_retries: int = 2,
) -> Literal["generate_answer", "rewrite_query", "web_search"]:
    """Route after document grading: answer if relevant, retry once if allowed, or fall back."""
    relevant = state.get("relevant_documents") or []
    if relevant:
        return "generate_answer"

    retry_count = int(state.get("retry_count", 0))
    chat_mode = state.get("chat_mode", "Auto (Agentic)")

    # Allow a single controlled retry with broadened rewrite before giving up or falling back
    if retry_count < min(max_retrieval_retries, 1):
        return "rewrite_query"

    if chat_mode == "Auto (Agentic)" and state.get("query_type") != "DOCUMENT_ONLY":
        return "web_search"

    return "generate_answer"


def route_after_validation(
    state: AgentState,
    max_validation_retries: int = 1,
) -> Literal["generate_answer", "end"]:
    """Allow at most one controlled regeneration attempt if validation fails."""
    passed = bool(state.get("validation_passed", True))
    retries = int(state.get("validation_retries", 0))
    if not passed and retries < max_validation_retries:
        return "generate_answer"
    return "end"
