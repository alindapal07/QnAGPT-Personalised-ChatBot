"""Typed LangGraph state definition for the Agentic RAG workflow."""

from typing import Annotated, Any, Dict, List
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
    retrieved_documents: List[Dict[str, Any]]
    relevant_documents: List[Dict[str, Any]]
    web_results: List[Dict[str, Any]]
    context: str
    answer: str
    sources: List[Dict[str, Any]]
    web_sources: List[Dict[str, str]]
    retry_count: int
    validation_retries: int
    tool_calls_count: int
    validation_passed: bool
    document_search_required: bool
    web_search_required: bool
    activity: str
    error: str
