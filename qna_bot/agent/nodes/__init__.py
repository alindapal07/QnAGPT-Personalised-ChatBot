"""Modular LangGraph node suite for RAG, Web Search, CRAG, MCP Tool Discovery, Permissions, and Execution."""

from langchain_groq import ChatGroq

from qna_bot.agent.nodes.base import (
    BaseNodeSuite,
    _QueryAnalysisSchema,
    _TaskPlanSchema,
    _ValidationSchema,
    _clean_chat_history,
    _collect_web_sources_from_messages,
    _deduplicate_doc_sources,
    _executed_call_signatures_in_turn,
    _extract_json_model,
    _format_document_context,
    _format_web_context,
    _is_tool_use_failed_error,
    _messages_for_response_llm,
    _normalize_agent_tool_calls,
    _recent_history_text,
    _requires_builtin_web_browse_loop,
    _select_relevant_tools_for_binding,
)
from qna_bot.agent.nodes.crag import CRAGNodesMixin
from qna_bot.agent.nodes.intent import IntentNodesMixin
from qna_bot.agent.nodes.mcp_exec import MCPNodesMixin
from qna_bot.agent.nodes.retrieval import RetrievalNodesMixin
from qna_bot.agent.nodes.synthesis import SynthesisNodesMixin
from qna_bot.agent.nodes.web import WebNodesMixin


class AgentNodeSuite(
    IntentNodesMixin,
    RetrievalNodesMixin,
    CRAGNodesMixin,
    WebNodesMixin,
    MCPNodesMixin,
    SynthesisNodesMixin,
    BaseNodeSuite,
):
    """Stateful node suite providing modular RAG, Web Search, CRAG, and Agentic MCP + HITL nodes."""
    pass


__all__ = [
    "AgentNodeSuite",
    "BaseNodeSuite",
    "ChatGroq",
    "CRAGNodesMixin",
    "IntentNodesMixin",
    "MCPNodesMixin",
    "RetrievalNodesMixin",
    "SynthesisNodesMixin",
    "WebNodesMixin",
    "_QueryAnalysisSchema",
    "_TaskPlanSchema",
    "_ValidationSchema",
    "_clean_chat_history",
    "_collect_web_sources_from_messages",
    "_deduplicate_doc_sources",
    "_executed_call_signatures_in_turn",
    "_extract_json_model",
    "_format_document_context",
    "_format_web_context",
    "_is_tool_use_failed_error",
    "_messages_for_response_llm",
    "_normalize_agent_tool_calls",
    "_recent_history_text",
    "_requires_builtin_web_browse_loop",
    "_select_relevant_tools_for_binding",
]
