"""Typed LangGraph state definition for the Agentic RAG & MCP Workspace."""

from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """Shared state passed across all LangGraph nodes and persisted via checkpointing."""

    messages: Annotated[List[AnyMessage], add_messages]
    user_query: str
    rewritten_query: str
    query_type: str
    chat_mode: str
    user_id: str
    thread_id: str
    system_prompt: str

    # RAG & Web Search state
    retrieved_documents: List[Dict[str, Any]]
    relevant_documents: List[Dict[str, Any]]
    web_results: List[Dict[str, Any]]
    opened_webpages: List[Dict[str, Any]]
    verified_research: Dict[str, Any]
    context: str
    answer: str
    sources: List[Dict[str, Any]]
    web_sources: List[Dict[str, str]]
    retry_count: int
    web_retry_count: int
    validation_retries: int
    tool_calls_count: int
    validation_passed: bool
    document_search_required: bool
    web_search_required: bool

    # Agentic MCP & HITL Approval state
    available_tools: List[Dict[str, Any]]
    planned_steps: List[str]
    pending_tool_calls: List[Dict[str, Any]]
    pending_approval: Optional[Dict[str, Any]]
    approval_decision: Optional[Dict[str, Any]]
    tool_executions: List[Dict[str, Any]]
    iteration_count: int
    max_iterations: int
    agent_status: str
    needs_more_steps: bool

    activity: str
    error: str
