"""Chat orchestration service combining LangGraph Agentic RAG + MCP + HITL, Vision fallback, and SQLite memory."""

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Dict, Generator, List, Mapping, Optional, Sequence

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from qna_bot.agent.approval import ApprovalManager
from qna_bot.agent.graph import build_agent_graph
from qna_bot.agent.memory import ConversationMemoryManager
from qna_bot.config.settings import AppSettings, get_settings
from qna_bot.core.logging_config import get_logger
from qna_bot.database.session import SQLAlchemyDatabase
from qna_bot.mcp.client_manager import MCPClientManager
from qna_bot.rag.grader import DocumentGrader
from qna_bot.rag.loaders import load_document_bytes
from qna_bot.rag.retriever import DocumentRetriever
from qna_bot.rag.vector_store import FAISSVectorStoreManager, VectorStoreBackend
from qna_bot.repositories import AuditAndTelemetryRepository, ConversationRepository
from qna_bot.services.agent_service import AIAgent
from qna_bot.tools.web_search import DuckDuckGoWebSearcher
import uuid

logger = get_logger(__name__)


@dataclass(frozen=True)
class ChatTurnResult:
    """Final or interrupted structured output of a single chat turn."""

    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    web_sources: List[Dict[str, str]] = field(default_factory=list)
    tool_executions: List[Dict[str, Any]] = field(default_factory=list)
    planned_steps: List[str] = field(default_factory=list)
    execution_trace: List[Dict[str, Any]] = field(default_factory=list)
    pending_approval: Optional[Dict[str, Any]] = None
    agent_status: str = "Completed"
    query_type: str = "GENERAL"
    error: str = ""


def _build_node_stage_event(
    node_name: str,
    update: Dict[str, Any],
    elapsed_s: float,
) -> Dict[str, Any]:
    """Convert an actual LangGraph node update into a concise, sanitized execution timeline event."""
    state = "failed" if update.get("error") and node_name in ("execute_tools", "generate_answer", "generate_response") else "completed"
    detail = ""
    if node_name == "analyze_query":
        label = "Intent classification"
        detail = str(update.get("query_type") or "")
    elif node_name == "rewrite_query":
        label = "Refining retrieval query"
    elif node_name == "retrieve_documents":
        label = "Searching Knowledge Base"
        detail = f"{len(update.get('retrieved_documents') or [])} chunks retrieved"
    elif node_name == "grade_documents":
        label = "Checking document relevance"
        detail = f"{len(update.get('relevant_documents') or [])} relevant"
    elif node_name == "web_search":
        label = "Web search"
        detail = f"{len(update.get('web_results') or [])} results found"
    elif node_name == "refine_web_search":
        label = "Refining search query"
    elif node_name == "open_webpages":
        label = "Retrieving webpage content"
        pages = update.get("opened_webpages") or []
        ok_cnt = sum(1 for p in pages if p.get("success"))
        fail_cnt = len(pages) - ok_cnt
        detail = f"{ok_cnt} pages read" + (f", {fail_cnt} inaccessible" if fail_cnt else "")
    elif node_name == "extract_and_verify":
        label = "Extracting & verifying information"
        comps = (update.get("verified_research") or {}).get("companies") or []
        detail = f"{len(comps)} structured entries" if comps else "Cross-checked sources"
    elif node_name == "discover_tools":
        label = "Verifying tools & MCP connections"
        detail = f"{len(update.get('available_tools') or [])} tools"
    elif node_name == "plan_task":
        label = "Planning execution"
        detail = f"{len(update.get('planned_steps') or [])} steps"
    elif node_name == "select_tools":
        calls = update.get("pending_tool_calls") or []
        if calls:
            label = "Tool selection"
            detail = ", ".join(str(c.get("name") or "") for c in calls[:3])
        else:
            label = "Preparing final synthesis"
    elif node_name == "check_permissions":
        label = "Checking tool permissions"
        detail = "Approval requested" if update.get("pending_approval") else "Authorized"
    elif node_name == "request_approval":
        label = "Approval decision"
        ok = bool((update.get("approval_decision") or {}).get("approved"))
        detail = "Approved" if ok else "Rejected"
    elif node_name in ("execute_tools", "tools"):
        label = "Executing tool"
        execs = update.get("tool_executions") or []
        if execs:
            last = execs[-1]
            srv = str(last.get("server_name") or "Tool")
            tname = str(last.get("tool_name") or "")
            dur = last.get("duration_ms")
            dur_str = f" ({dur:.0f} ms)" if isinstance(dur, (int, float)) else ""
            detail = f"{srv} :: {tname}{dur_str}"
            if last.get("status") in ("failed", "blocked", "rejected"):
                state = "failed"
    elif node_name == "observe_results":
        label = "Observing tool output"
    elif node_name == "evaluate_progress":
        label = "Evaluating progress"
        detail = "Next iteration" if update.get("needs_more_steps") else "Ready for final response"
    elif node_name in ("generate_answer", "generate_response", "direct_answer"):
        label = "Executing built-in tool" if update.get("tool_calls_count") and not update.get("answer") else "Generating response"
    elif node_name == "validate_answer":
        label = "Validating citations"
        detail = "Verified" if update.get("validation_passed", True) else "Refining"
    else:
        label = node_name.replace("_", " ").title()

    return {
        "node": node_name,
        "label": label,
        "detail": detail,
        "state": state,
        "elapsed_s": round(elapsed_s, 2),
    }


class ChatService:
    """Coordinates LangGraph execution, HITL interrupts/resumes, multimodal vision, and SQLite threads."""

    def __init__(
        self,
        settings: Optional[AppSettings] = None,
        memory_manager: Optional[ConversationMemoryManager] = None,
        vector_store: Optional[VectorStoreBackend] = None,
        web_searcher: Optional[DuckDuckGoWebSearcher] = None,
        mcp_manager: Optional[MCPClientManager] = None,
        approval_manager: Optional[ApprovalManager] = None,
        db: Optional[SQLAlchemyDatabase] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.db = db
        self.conv_repo = ConversationRepository(db) if db is not None else None
        self.audit_repo = AuditAndTelemetryRepository(db) if db is not None else None
        self.memory = memory_manager or ConversationMemoryManager(
            db_path=self.settings.langgraph_checkpoint_db
        )
        self.vector_store: VectorStoreBackend = vector_store or FAISSVectorStoreManager(
            base_dir=self.settings.vector_store_dir
        )
        self.web_searcher = web_searcher or DuckDuckGoWebSearcher()
        self.mcp_manager = mcp_manager
        self.approval_manager = approval_manager or (
            ApprovalManager(db_manager=mcp_manager.db) if mcp_manager is not None else None
        )

    def create_thread(
        self,
        user_id: str,
        title: str = "New Conversation",
        chat_mode: str = "Auto (Agentic)",
        thread_id: Optional[str] = None,
    ) -> str:
        tid = self.memory.create_new_thread(
            user_id=user_id,
            title=title,
            chat_mode=chat_mode,
            thread_id=thread_id,
        )
        if self.conv_repo is not None:
            try:
                self.conv_repo.create(
                    user_id=user_id,
                    title=title,
                    agent_mode=chat_mode,
                    conversation_id=tid,
                )
            except Exception:
                pass
        return tid

    def list_threads(self, user_id: str, search_query: str = "") -> List[Dict[str, Any]]:
        return self.memory.retrieve_all_threads(user_id=user_id, search_query=search_query)

    def load_thread_history(self, thread_id: str, user_id: str) -> List[Dict[str, Any]]:
        return self.memory.retrieve_chat_history(thread_id=thread_id, user_id=user_id)

    def delete_thread(self, thread_id: str, user_id: str) -> bool:
        ok = self.memory.delete_thread(thread_id=thread_id, user_id=user_id)
        if self.conv_repo is not None:
            try:
                self.conv_repo.delete(thread_id, user_id)
            except Exception:
                pass
        return ok

    def persist_message(
        self,
        thread_id: str,
        user_id: str,
        message: Dict[str, Any],
        chat_mode: Optional[str] = None,
    ) -> None:
        self.memory.append_message(thread_id=thread_id, user_id=user_id, message=message)
        if message.get("role") == "user":
            self.memory.update_thread_title(
                thread_id=thread_id,
                user_id=user_id,
                first_prompt=str(message.get("content", "")),
                chat_mode=chat_mode,
            )
        if self.conv_repo is not None:
            try:
                self.conv_repo.append_message(
                    conversation_id=thread_id,
                    user_id=user_id,
                    role=str(message.get("role", "user")),
                    content=str(message.get("content", "")),
                    message_type="image" if message.get("image_b64") or message.get("images") else "text",
                    metadata={
                        "sources": message.get("sources") or [],
                        "web_sources": message.get("web_sources") or [],
                        "tool_events": message.get("tool_events") or [],
                    },
                )
            except Exception:
                pass

    def sync_thread_messages(
        self,
        thread_id: str,
        user_id: str,
        messages: List[Dict[str, Any]],
    ) -> None:
        self.memory.replace_thread_messages(
            thread_id=thread_id,
            user_id=user_id,
            messages=messages,
        )

    def extract_inline_document_context(
        self,
        files: Sequence[tuple[str, bytes]],
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Parse temporary inline document attachments without persisting into FAISS."""
        excerpts: List[str] = []
        inline_sources: List[Dict[str, Any]] = []
        for fname, raw_bytes in files:
            docs = load_document_bytes(
                file_bytes=raw_bytes,
                filename=fname,
                max_size_mb=self.settings.max_document_size_mb,
                allowed_types=self.settings.allowed_document_types,
            )
            combined = "\n".join(d.page_content for d in docs)[:6000]
            first_page = docs[0].metadata.get("page") if docs else None
            excerpts.append(f"[Attached File: {fname}]\n{combined}")
            inline_sources.append(
                {
                    "source": fname,
                    "page": first_page,
                    "score": 1.0,
                    "excerpt": combined[:320],
                }
            )
        return "\n\n".join(excerpts), inline_sources

    def stream_vision_turn(
        self,
        messages: Sequence[Mapping[str, Any]],
        text_model: str,
        vision_model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str,
    ) -> Generator[str, None, None]:
        """Delegate multimodal image turns to the existing ChatGroq vision pipeline."""
        agent = AIAgent(
            text_model=text_model,
            vision_model=vision_model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=self.settings.groq_api_key,
            system_prompt=system_prompt,
            stream_delay_sec=self.settings.stream_chunk_delay_sec,
        )
        yield from agent.stream(list(messages), has_image=True)

    def _build_graph(
        self,
        user_id: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
    ) -> Any:
        retriever = DocumentRetriever(vector_store=self.vector_store, settings=self.settings)
        grader = DocumentGrader(
            model_name=model_name,
            api_key=self.settings.groq_api_key,
            settings=self.settings,
        )
        return build_agent_graph(
            user_id=user_id,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=self.settings.groq_api_key,
            retriever=retriever,
            grader=grader,
            web_searcher=self.web_searcher,
            memory_manager=self.memory,
            mcp_manager=self.mcp_manager,
            approval_manager=self.approval_manager,
            settings=self.settings,
            use_cache=self.mcp_manager is not None,
        )

    def _consume_stream(
        self,
        graph: Any,
        graph_input: Any,
        thread_id: str,
        on_activity: Optional[Callable[[str], None]] = None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> ChatTurnResult:
        config = {"configurable": {"thread_id": thread_id}}
        final_state: Dict[str, Any] = {}
        pending_interrupt: Optional[Dict[str, Any]] = None
        trace: List[Dict[str, Any]] = []
        started = time.perf_counter()

        for event in graph.stream(graph_input, config=config, stream_mode="updates"):
            if "__interrupt__" in event:
                interrupts = event["__interrupt__"]
                if interrupts:
                    first_int = interrupts[0]
                    val = getattr(first_int, "value", first_int)
                    if isinstance(val, dict):
                        pending_interrupt = dict(val)
                continue

            for node_name, update in event.items():
                if not isinstance(update, dict):
                    continue
                final_state.update(update)
                elapsed_s = time.perf_counter() - started
                stage = _build_node_stage_event(node_name, update, elapsed_s=elapsed_s)
                trace.append(stage)
                activity = update.get("activity")
                if activity and on_activity:
                    on_activity(str(activity))
                if on_event:
                    on_event({"stage": stage, "trace": list(trace), "elapsed_s": elapsed_s})
                logger.debug("Graph node '%s' completed (thread=%s)", node_name, thread_id)

        if pending_interrupt is not None:
            return ChatTurnResult(
                answer="",
                sources=list(final_state.get("sources") or []),
                web_sources=list(final_state.get("web_sources") or []),
                tool_executions=list(final_state.get("tool_executions") or []),
                planned_steps=list(final_state.get("planned_steps") or []),
                execution_trace=trace,
                pending_approval=pending_interrupt,
                agent_status="Waiting for approval",
                query_type=str(final_state.get("query_type") or "AGENT_TASK"),
                error=str(final_state.get("error") or ""),
            )

        return ChatTurnResult(
            answer=str(final_state.get("answer") or ""),
            sources=list(final_state.get("sources") or []),
            web_sources=list(final_state.get("web_sources") or []),
            tool_executions=list(final_state.get("tool_executions") or []),
            planned_steps=list(final_state.get("planned_steps") or []),
            execution_trace=trace,
            pending_approval=None,
            agent_status=str(final_state.get("agent_status") or "Completed"),
            query_type=str(final_state.get("query_type") or "GENERAL"),
            error=str(final_state.get("error") or ""),
        )

    @staticmethod
    def _scoped_checkpoint_id(user_id: str, thread_id: str) -> str:
        prefix = f"{user_id}:"
        return thread_id if thread_id.startswith(prefix) else f"{prefix}{thread_id}"

    def _record_execution_telemetry(
        self,
        user_id: str,
        thread_id: str,
        execution_id: str,
        result: ChatTurnResult,
    ) -> None:
        if self.conv_repo is not None:
            try:
                last_step = result.execution_trace[-1]["node"] if result.execution_trace else ""
                status = (
                    "waiting_approval"
                    if result.pending_approval
                    else ("failed" if result.error else "completed")
                )
                self.conv_repo.finish_execution(
                    user_id=user_id,
                    execution_id=execution_id,
                    status=status,
                    current_step=last_step,
                    metadata={
                        "query_type": result.query_type,
                        "stages": len(result.execution_trace),
                        "sources": len(result.sources),
                        "web_sources": len(result.web_sources),
                    },
                )
            except Exception:
                pass
        if self.audit_repo is not None:
            for item in result.tool_executions:
                try:
                    self.audit_repo.log_mcp_execution(
                        user_id=user_id,
                        conversation_id=thread_id,
                        server_id=item.get("server_id") or None,
                        tool_name=str(item.get("tool_name") or "tool"),
                        status=str(item.get("status") or "completed"),
                        execution_time=float(item.get("duration_ms") or 0.0) / 1000.0,
                        error_message=str(item.get("error") or ""),
                    )
                except Exception:
                    pass

    def run_agentic_turn(
        self,
        user_query: str,
        thread_id: str,
        user_id: str,
        chat_mode: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str,
        on_activity: Optional[Callable[[str], None]] = None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
        max_iterations: Optional[int] = None,
    ) -> ChatTurnResult:
        """Execute the compiled LangGraph Agentic RAG & MCP graph with live step updates."""
        self.memory.ensure_thread(
            thread_id=thread_id,
            user_id=user_id,
            chat_mode=chat_mode,
        )
        execution_id = f"exec_{uuid.uuid4().hex[:14]}"
        if self.conv_repo is not None:
            try:
                self.conv_repo.create(
                    user_id=user_id,
                    agent_mode=chat_mode,
                    model_name=model_name,
                    conversation_id=thread_id,
                )
                self.conv_repo.start_execution(
                    user_id=user_id,
                    conversation_id=thread_id,
                    execution_id=execution_id,
                    metadata={"chat_mode": chat_mode, "model_name": model_name},
                )
            except Exception:
                pass

        graph = self._build_graph(
            user_id=user_id,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        input_state = {
            "messages": [HumanMessage(content=user_query)],
            "user_query": user_query,
            "chat_mode": chat_mode,
            "user_id": user_id,
            "thread_id": thread_id,
            "system_prompt": system_prompt,
            "max_iterations": max_iterations or self.settings.max_agent_iterations,
        }
        result = self._consume_stream(
            graph,
            input_state,
            thread_id=self._scoped_checkpoint_id(user_id, thread_id),
            on_activity=on_activity,
            on_event=on_event,
        )
        self._record_execution_telemetry(user_id, thread_id, execution_id, result)
        return result

    def resume_approval(
        self,
        thread_id: str,
        user_id: str,
        approval_id: str,
        approved: bool,
        model_name: str,
        temperature: float,
        max_tokens: int,
        on_activity: Optional[Callable[[str], None]] = None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> ChatTurnResult:
        """Resume a paused LangGraph execution after recording backend approval decision."""
        if self.approval_manager is not None:
            self.approval_manager.decide(
                approval_id=approval_id,
                user_id=user_id,
                approved=approved,
            )
        execution_id = f"exec_{uuid.uuid4().hex[:14]}"
        if self.conv_repo is not None:
            try:
                self.conv_repo.start_execution(
                    user_id=user_id,
                    conversation_id=thread_id,
                    execution_id=execution_id,
                    current_step="request_approval",
                    metadata={"approval_id": approval_id, "approved": approved},
                )
            except Exception:
                pass

        graph = self._build_graph(
            user_id=user_id,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        result = self._consume_stream(
            graph,
            Command(resume={"approved": approved, "approval_id": approval_id}),
            thread_id=self._scoped_checkpoint_id(user_id, thread_id),
            on_activity=on_activity,
            on_event=on_event,
        )
        self._record_execution_telemetry(user_id, thread_id, execution_id, result)
        return result

    def stream_text_tokens(self, text: str) -> Generator[str, None, None]:
        """Stream a completed response smoothly by word tokens for Streamlit write_stream."""
        delay = self.settings.stream_chunk_delay_sec
        tokens = text.split(" ")
        for idx, word in enumerate(tokens):
            if delay > 0:
                time.sleep(min(delay, 0.015))
            yield word if idx == len(tokens) - 1 else f"{word} "
