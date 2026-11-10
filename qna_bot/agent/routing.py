"""Deterministic conditional routing logic for the LangGraph Agentic RAG & MCP workflow."""

from typing import Literal

from qna_bot.agent.state import AgentState
from qna_bot.tools.web_research import needs_structured_entity_research


def route_after_analysis(
    state: AgentState,
) -> Literal["direct_answer", "rewrite_query", "web_search", "discover_tools"]:
    """Deterministically select the next graph node based on chat mode and classified intent."""
    chat_mode = state.get("chat_mode", "Auto (Agentic)")
    query_type = (state.get("query_type") or "GENERAL").upper()
    doc_required = bool(state.get("document_search_required", False))
    web_required = bool(state.get("web_search_required", False))

    if chat_mode == "General Chat":
        return "direct_answer"

    if chat_mode == "Document Chat":
        return "rewrite_query"

    if chat_mode == "Agent Workspace" or query_type == "AGENT_TASK":
        return "discover_tools"

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


def route_after_permissions(
    state: AgentState,
) -> Literal["request_approval", "execute_tools", "observe_results"]:
    """Route tool calls to HITL approval, execution, or straight to observation if none/blocked."""
    if state.get("pending_approval"):
        return "request_approval"
    if state.get("pending_tool_calls"):
        return "execute_tools"
    return "observe_results"


def route_after_approval(
    state: AgentState,
) -> Literal["execute_tools", "observe_results"]:
    """Proceed to tool execution only if backend approval was granted."""
    decision = state.get("approval_decision") or {}
    if decision.get("approved"):
        return "execute_tools"
    return "observe_results"


def route_after_progress(
    state: AgentState,
) -> Literal["select_tools", "generate_response"]:
    """Continue multi-step agent loop if needed and within max_iterations."""
    needs_more = bool(state.get("needs_more_steps", False))
    iteration = int(state.get("iteration_count", 0))
    max_iter = int(state.get("max_iterations", 10))
    if needs_more and iteration < max_iter:
        return "select_tools"
    return "generate_response"


def route_after_web_search(
    state: AgentState,
    max_web_retries: int = 1,
) -> Literal["open_webpages", "refine_web_search", "generate_answer"]:
    """Open discovered search hits, retry search once if empty, or answer cleanly."""
    if state.get("web_results"):
        return "open_webpages"
    if int(state.get("web_retry_count", 0)) < max_web_retries:
        return "refine_web_search"
    return "generate_answer"


def route_after_web_extraction(
    state: AgentState,
    max_web_retries: int = 1,
) -> Literal["extract_and_verify", "refine_web_search", "generate_answer"]:
    """Route to structured verification when entity details are needed, retry once if pages failed, or synthesize."""
    opened = state.get("opened_webpages") or []
    any_opened = any(bool(p.get("success")) and bool(p.get("content")) for p in opened)
    query = str(state.get("user_query") or "")
    structured_needed = needs_structured_entity_research(query)

    if any_opened:
        if structured_needed:
            return "extract_and_verify"
        return "generate_answer"

    if structured_needed and int(state.get("web_retry_count", 0)) < max_web_retries:
        return "refine_web_search"

    return "generate_answer"
