"""LangGraph StateGraph assembly for Agentic RAG, Web Search, and Dynamic MCP Execution."""

from functools import partial
import threading
from typing import Any, Dict, Optional, Tuple

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from qna_bot.agent.approval import ApprovalManager
from qna_bot.agent.memory import ConversationMemoryManager
from qna_bot.agent.nodes import AgentNodeSuite
from qna_bot.agent.routing import (
    route_after_analysis,
    route_after_approval,
    route_after_grading,
    route_after_permissions,
    route_after_progress,
    route_after_validation,
    route_after_web_extraction,
    route_after_web_search,
)
from qna_bot.agent.state import AgentState
from qna_bot.config.settings import AppSettings, get_settings
from qna_bot.database.repositories import AuditLogRepository
from qna_bot.mcp.client_manager import MCPClientManager
from qna_bot.rag.grader import DocumentGrader
from qna_bot.rag.retriever import DocumentRetriever
from qna_bot.rag.vector_store import FAISSVectorStoreManager
from qna_bot.tools.document_tools import build_document_tools
from qna_bot.tools.web_search import (
    DuckDuckGoWebSearcher,
    build_open_web_page_tool,
    build_web_search_tool,
)

_GRAPH_CACHE_LOCK = threading.RLock()
_COMPILED_GRAPH_CACHE: Dict[Tuple[Any, ...], Any] = {}


def clear_compiled_graph_cache() -> None:
    """Clear cached compiled graphs (used when restarting services or in tests)."""
    with _GRAPH_CACHE_LOCK:
        _COMPILED_GRAPH_CACHE.clear()


def build_agent_graph(
    user_id: str,
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    api_key: Optional[str] = None,
    retriever: Optional[DocumentRetriever] = None,
    grader: Optional[DocumentGrader] = None,
    web_searcher: Optional[DuckDuckGoWebSearcher] = None,
    memory_manager: Optional[ConversationMemoryManager] = None,
    mcp_manager: Optional[MCPClientManager] = None,
    approval_manager: Optional[ApprovalManager] = None,
    settings: Optional[AppSettings] = None,
    use_cache: bool = False,
) -> Any:
    """Compile the LangGraph StateGraph with RAG, Web Search, MCP tools, HITL approval, and SQLite checkpointing."""
    cfg = settings or get_settings()
    resolved_key = (api_key if api_key is not None else cfg.groq_api_key).strip()
    resolved_model = model_name or cfg.default_text_model

    if use_cache and mcp_manager is not None:
        tool_sig = tuple(
            (t.namespaced_name, t.permission_category, t.auto_approve, t.enabled)
            for t in mcp_manager.list_available_tools(user_id=user_id)
        )
        cache_key = (
            user_id,
            resolved_model,
            round(float(temperature), 3),
            int(max_tokens),
            str(cfg.sqlite_db_path),
            str(cfg.vector_store_dir),
            tool_sig,
        )
        with _GRAPH_CACHE_LOCK:
            cached = _COMPILED_GRAPH_CACHE.get(cache_key)
            if cached is not None:
                return cached

    active_retriever = retriever or DocumentRetriever(
        vector_store=FAISSVectorStoreManager(base_dir=cfg.vector_store_dir),
        settings=cfg,
    )
    active_grader = grader or DocumentGrader(
        model_name=resolved_model,
        api_key=resolved_key,
        settings=cfg,
    )
    active_searcher = web_searcher or DuckDuckGoWebSearcher()
    active_memory = memory_manager or ConversationMemoryManager(db_path=cfg.sqlite_db_path)

    builtin_tools = [
        *build_document_tools(active_retriever, user_id=user_id),
        build_web_search_tool(active_searcher),
        build_open_web_page_tool(active_searcher),
    ]

    nodes = AgentNodeSuite(
        retriever=active_retriever,
        grader=active_grader,
        web_searcher=active_searcher,
        tools=builtin_tools,
        model_name=resolved_model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=resolved_key,
        settings=cfg,
        mcp_manager=mcp_manager,
        approval_manager=approval_manager,
        audit_repo=AuditLogRepository(mcp_manager.db) if mcp_manager is not None else None,
    )

    workflow = StateGraph(AgentState)

    # RAG & General Chat nodes
    workflow.add_node("analyze_query", nodes.analyze_query)
    workflow.add_node("rewrite_query", nodes.rewrite_query)
    workflow.add_node("retrieve_documents", nodes.retrieve_documents)
    workflow.add_node("grade_documents", nodes.grade_documents)
    workflow.add_node("web_search", nodes.web_search)
    workflow.add_node("refine_web_search", nodes.refine_web_search)
    workflow.add_node("open_webpages", nodes.open_webpages)
    workflow.add_node("extract_and_verify", nodes.extract_and_verify)
    workflow.add_node("generate_answer", nodes.generate_answer)
    workflow.add_node("direct_answer", nodes.direct_answer)
    workflow.add_node("validate_answer", nodes.validate_answer)
    workflow.add_node("tools", ToolNode(builtin_tools))

    # Agentic MCP + HITL Approval nodes
    workflow.add_node("discover_tools", nodes.discover_tools)
    workflow.add_node("plan_task", nodes.plan_task)
    workflow.add_node("select_tools", nodes.select_tools)
    workflow.add_node("check_permissions", nodes.check_permissions)
    workflow.add_node("request_approval", nodes.request_approval)
    workflow.add_node("execute_tools", nodes.execute_tools)
    workflow.add_node("observe_results", nodes.observe_results)
    workflow.add_node("evaluate_progress", nodes.evaluate_progress)
    workflow.add_node("generate_response", nodes.generate_response)

    workflow.add_edge(START, "analyze_query")

    workflow.add_conditional_edges(
        "analyze_query",
        route_after_analysis,
        {
            "direct_answer": "direct_answer",
            "rewrite_query": "rewrite_query",
            "web_search": "web_search",
            "discover_tools": "discover_tools",
        },
    )

    # General / Built-in ToolNode path
    workflow.add_conditional_edges(
        "direct_answer",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )
    workflow.add_edge("tools", "direct_answer")

    # Document RAG + Web Search path
    workflow.add_edge("rewrite_query", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        partial(route_after_grading, max_retrieval_retries=cfg.max_retrieval_retries),
        {
            "generate_answer": "generate_answer",
            "rewrite_query": "rewrite_query",
            "web_search": "web_search",
        },
    )

    workflow.add_conditional_edges(
        "web_search",
        partial(route_after_web_search, max_web_retries=1),
        {
            "open_webpages": "open_webpages",
            "refine_web_search": "refine_web_search",
            "generate_answer": "generate_answer",
        },
    )
    workflow.add_edge("refine_web_search", "web_search")
    workflow.add_conditional_edges(
        "open_webpages",
        partial(route_after_web_extraction, max_web_retries=1),
        {
            "extract_and_verify": "extract_and_verify",
            "refine_web_search": "refine_web_search",
            "generate_answer": "generate_answer",
        },
    )
    workflow.add_edge("extract_and_verify", "generate_answer")
    workflow.add_edge("generate_answer", "validate_answer")

    workflow.add_conditional_edges(
        "validate_answer",
        partial(route_after_validation, max_validation_retries=cfg.max_validation_retries),
        {
            "generate_answer": "generate_answer",
            "end": END,
        },
    )

    # Agentic MCP + HITL Approval loop
    workflow.add_edge("discover_tools", "plan_task")
    workflow.add_edge("plan_task", "select_tools")
    workflow.add_edge("select_tools", "check_permissions")

    workflow.add_conditional_edges(
        "check_permissions",
        route_after_permissions,
        {
            "request_approval": "request_approval",
            "execute_tools": "execute_tools",
            "observe_results": "observe_results",
        },
    )

    workflow.add_conditional_edges(
        "request_approval",
        route_after_approval,
        {
            "execute_tools": "execute_tools",
            "observe_results": "observe_results",
        },
    )

    workflow.add_edge("execute_tools", "observe_results")
    workflow.add_edge("observe_results", "evaluate_progress")

    workflow.add_conditional_edges(
        "evaluate_progress",
        route_after_progress,
        {
            "select_tools": "select_tools",
            "generate_response": "generate_response",
        },
    )

    workflow.add_edge("generate_response", END)

    compiled = workflow.compile(checkpointer=active_memory.checkpointer)
    if use_cache and mcp_manager is not None:
        with _GRAPH_CACHE_LOCK:
            if len(_COMPILED_GRAPH_CACHE) >= 32:
                _COMPILED_GRAPH_CACHE.clear()
            _COMPILED_GRAPH_CACHE[cache_key] = compiled
    return compiled
