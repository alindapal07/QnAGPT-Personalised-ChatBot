"""LangGraph agent package exports."""

from typing import Any, Dict, List, Tuple

from qna_bot import resolve_lazy_export

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "AgentNodeSuite": ("qna_bot.agent.nodes", "AgentNodeSuite"),
    "AgentState": ("qna_bot.agent.state", "AgentState"),
    "ApprovalManager": ("qna_bot.agent.approval", "ApprovalManager"),
    "ConversationMemoryManager": ("qna_bot.agent.memory", "ConversationMemoryManager"),
    "build_agent_graph": ("qna_bot.agent.graph", "build_agent_graph"),
    "build_mcp_langchain_tools": ("qna_bot.agent.mcp_adapter", "build_mcp_langchain_tools"),
    "build_pydantic_args_schema": ("qna_bot.agent.mcp_adapter", "build_pydantic_args_schema"),
    "classify_time_bucket": ("qna_bot.agent.memory", "classify_time_bucket"),
    "clear_compiled_graph_cache": ("qna_bot.agent.graph", "clear_compiled_graph_cache"),
    "compute_arguments_hash": ("qna_bot.agent.approval", "compute_arguments_hash"),
    "create_new_thread": ("qna_bot.agent.memory", "create_new_thread"),
    "retrieve_all_threads": ("qna_bot.agent.memory", "retrieve_all_threads"),
    "retrieve_chat_history": ("qna_bot.agent.memory", "retrieve_chat_history"),
    "route_after_analysis": ("qna_bot.agent.routing", "route_after_analysis"),
    "route_after_approval": ("qna_bot.agent.routing", "route_after_approval"),
    "route_after_grading": ("qna_bot.agent.routing", "route_after_grading"),
    "route_after_permissions": ("qna_bot.agent.routing", "route_after_permissions"),
    "route_after_progress": ("qna_bot.agent.routing", "route_after_progress"),
    "route_after_validation": ("qna_bot.agent.routing", "route_after_validation"),
    "validate_tool_arguments": ("qna_bot.agent.mcp_adapter", "validate_tool_arguments"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    return resolve_lazy_export(__name__, globals(), _EXPORTS, name)


def __dir__() -> List[str]:
    return sorted(__all__)
