"""LangGraph StateGraph assembly for the Agentic RAG Assistant."""

from functools import partial
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from qna_bot.agent.memory import ConversationMemoryManager
from qna_bot.agent.nodes import AgentNodeSuite
from qna_bot.agent.routing import (
    route_after_analysis,
    route_after_grading,
    route_after_validation,
)
from qna_bot.agent.state import AgentState
from qna_bot.config.settings import AppSettings, get_settings
from qna_bot.rag.grader import DocumentGrader
from qna_bot.rag.retriever import DocumentRetriever
from qna_bot.rag.vector_store import FAISSVectorStoreManager
from qna_bot.tools.document_tools import build_document_tools
from qna_bot.tools.web_search import DuckDuckGoWebSearcher, build_web_search_tool


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
    settings: Optional[AppSettings] = None,
) -> Any:
    """Compile the complete LangGraph Agentic RAG StateGraph with SQLite checkpointing."""
    cfg = settings or get_settings()
    resolved_key = (api_key if api_key is not None else cfg.groq_api_key).strip()
    resolved_model = model_name or cfg.default_text_model

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

    tools = [
        *build_document_tools(active_retriever, user_id=user_id),
        build_web_search_tool(active_searcher),
    ]

    nodes = AgentNodeSuite(
        retriever=active_retriever,
        grader=active_grader,
        web_searcher=active_searcher,
        tools=tools,
        model_name=resolved_model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=resolved_key,
        settings=cfg,
    )

    workflow = StateGraph(AgentState)

    workflow.add_node("analyze_query", nodes.analyze_query)
    workflow.add_node("rewrite_query", nodes.rewrite_query)
    workflow.add_node("retrieve_documents", nodes.retrieve_documents)
    workflow.add_node("grade_documents", nodes.grade_documents)
    workflow.add_node("web_search", nodes.web_search)
    workflow.add_node("generate_answer", nodes.generate_answer)
    workflow.add_node("direct_answer", nodes.direct_answer)
    workflow.add_node("validate_answer", nodes.validate_answer)
    workflow.add_node("tools", ToolNode(tools))

    workflow.add_edge(START, "analyze_query")

    workflow.add_conditional_edges(
        "analyze_query",
        route_after_analysis,
        {
            "direct_answer": "direct_answer",
            "rewrite_query": "rewrite_query",
            "web_search": "web_search",
        },
    )

    workflow.add_conditional_edges(
        "direct_answer",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )
    workflow.add_edge("tools", "direct_answer")

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

    workflow.add_edge("web_search", "generate_answer")
    workflow.add_edge("generate_answer", "validate_answer")

    workflow.add_conditional_edges(
        "validate_answer",
        partial(route_after_validation, max_validation_retries=cfg.max_validation_retries),
        {
            "generate_answer": "generate_answer",
            "end": END,
        },
    )

    return workflow.compile(checkpointer=active_memory.checkpointer)
