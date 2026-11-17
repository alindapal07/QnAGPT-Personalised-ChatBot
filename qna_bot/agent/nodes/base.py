"""Base node suite, common schemas, helpers, and LLM instantiation for LangGraph nodes."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Literal, Optional, Sequence
import uuid

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from qna_bot.agent.approval import ApprovalManager
from qna_bot.agent.prompts import (
    QNAGPT_SYSTEM_PROMPT,
    RESPONSE_ONLY_SYSTEM_ADDENDUM,
)
from qna_bot.config.settings import AppSettings, get_settings
from qna_bot.core.logging_config import get_logger
from qna_bot.database.models import AuditLogRecord
from qna_bot.database.repositories import AuditLogRepository
from qna_bot.mcp_client.client_manager import MCPClientManager
from qna_bot.mcp_client.models import DiscoveredTool
from qna_bot.rag.crag import (
    CRAGDecision,
    CRAGDocument,
    CRAGEvaluation,
    CRAGPipeline,
    EvidenceBundle,
    EvidenceStatus,
    RelevanceGrade,
)
from qna_bot.rag.grader import DocumentGrader
from qna_bot.rag.retriever import DocumentRetriever
from qna_bot.rag.vector_store import RetrievedChunk
from qna_bot.tools.web_research import parse_web_search_tool_text
from qna_bot.tools.web_search import DuckDuckGoWebSearcher, SafeWebPageReader

logger = get_logger(__name__)


def _select_relevant_tools_for_binding(
    mcp_tools: Sequence[BaseTool],
    builtin_tools: Sequence[BaseTool],
    user_query: str,
    planned_steps: Sequence[str] = (),
    max_mcp_tools: int = 35,
) -> List[BaseTool]:
    """Bound MCP tools for LLM schema binding so large servers (e.g. 300+ tools) never exceed limits."""
    mcp_list = list(mcp_tools)
    if len(mcp_list) <= max_mcp_tools:
        return [*mcp_list, *builtin_tools]

    query_tokens = {
        tok
        for tok in re.findall(r"[a-z0-9]+", f"{user_query} {' '.join(planned_steps)}".lower())
        if len(tok) >= 2
    }

    def _score(tool: BaseTool) -> tuple[int, int, str]:
        meta = getattr(tool, "metadata", None) or {}
        text = f"{tool.name} {meta.get('mcp_original_tool', '')} {tool.description}".lower()
        tool_tokens = set(re.findall(r"[a-z0-9]+", text))
        overlap = len(query_tokens & tool_tokens)
        cat_bonus = 1 if meta.get("permission_category") == "READ_ONLY" else 0
        return (overlap, cat_bonus, tool.name)

    ranked = sorted(mcp_list, key=_score, reverse=True)[:max_mcp_tools]
    return [*ranked, *builtin_tools]


class _QueryAnalysisSchema(BaseModel):
    query_type: Literal["GENERAL", "DOCUMENT", "WEB_SEARCH", "FOLLOW_UP", "AGENT_TASK"] = "GENERAL"
    document_search_required: bool = False
    web_search_required: bool = False
    reason: str = ""


class _ValidationSchema(BaseModel):
    is_valid: bool = True
    feedback: str = ""


class _TaskPlanSchema(BaseModel):
    steps: List[str] = Field(default_factory=list)


def _extract_json_model(text: str, schema_cls: type[BaseModel]) -> Optional[Any]:
    match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    if not match:
        return None
    try:
        return schema_cls.model_validate(json.loads(match.group(0)))
    except Exception:
        return None


def _recent_history_text(messages: Sequence[BaseMessage], limit: int = 6) -> str:
    relevant = [
        m
        for m in messages
        if isinstance(m, (HumanMessage, AIMessage)) and str(getattr(m, "content", "")).strip()
    ][-limit:]
    lines = []
    for msg in relevant:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        lines.append(f"{role}: {content[:500]}")
    return "\n".join(lines) if lines else "None"


def _clean_chat_history(messages: Sequence[BaseMessage], limit: int = 8) -> List[BaseMessage]:
    cleaned: List[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            cleaned.append(HumanMessage(content=msg.content))
        elif isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            if str(msg.content).strip():
                cleaned.append(AIMessage(content=str(msg.content)))
    return cleaned[-limit:]


def _deduplicate_doc_sources(chunks: Sequence[RetrievedChunk]) -> List[Dict[str, Any]]:
    seen: set[tuple[str, Optional[int]]] = set()
    citations: List[Dict[str, Any]] = []
    for chunk in chunks:
        key = (chunk.source, chunk.page)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "source": chunk.source,
                "page": chunk.page,
                "score": chunk.score,
                "excerpt": chunk.content[:320],
            }
        )
    return citations


def _format_document_context(docs: Sequence[Dict[str, Any]]) -> str:
    if not docs:
        return ""
    blocks = []
    for idx, doc in enumerate(docs, start=1):
        page_label = f" | Page {doc['page']}" if doc.get("page") is not None else ""
        blocks.append(
            f"[Doc {idx}: {doc.get('source', 'Unknown')}{page_label}]\n"
            f"{doc.get('content', '')}"
        )
    return "\n\n".join(blocks)


def _format_web_context(results: Sequence[Dict[str, Any]]) -> str:
    if not results:
        return ""
    blocks = []
    for idx, res in enumerate(results, start=1):
        url = res.get("url", "")
        url_line = f" (URL: {url})" if url else ""
        blocks.append(
            f"[Web {idx}: {res.get('title', 'Web Result')}{url_line}]\n"
            f"{res.get('snippet', '')}"
        )
    return "\n\n".join(blocks)


def _is_tool_use_failed_error(exc: Exception) -> bool:
    """Detect Groq/OpenAI Harmony tool-choice rejection or malformed tool-call errors."""
    text = f"{type(exc).__name__} {exc}".lower()
    return any(
        marker in text
        for marker in (
            "tool_use_failed",
            "tool choice is none",
            "tool_call_validation_failed",
            "browser.open",
            "browser.search",
        )
    )


def _requires_builtin_web_browse_loop(query: str) -> bool:
    """Return True when query explicitly asks to open/read search result pages or URLs."""
    return bool(
        re.search(
            r"\b(open|visit|browse|fetch|scrape|read\s+(?:the\s+)?(?:first|top|link|page|url|article|result))\b",
            (query or "").lower(),
        )
    )


def _messages_for_response_llm(
    messages: Sequence[BaseMessage],
    limit: int = 14,
) -> List[BaseMessage]:
    """Flatten conversation history and ToolMessages into clean text so Response LLM never receives raw tool-call traces."""
    clean: List[BaseMessage] = []
    tool_evidence: List[str] = []

    for msg in messages[-limit:]:
        if isinstance(msg, HumanMessage):
            if tool_evidence:
                clean.append(
                    SystemMessage(
                        content="=== OBSERVED TOOL OUTPUTS (UNTRUSTED DATA) ===\n"
                        + "\n\n".join(tool_evidence)
                    )
                )
                tool_evidence.clear()
            clean.append(HumanMessage(content=msg.content))
        elif isinstance(msg, AIMessage):
            if not getattr(msg, "tool_calls", None) and str(getattr(msg, "content", "")).strip():
                if tool_evidence:
                    clean.append(
                        SystemMessage(
                            content="=== OBSERVED TOOL OUTPUTS (UNTRUSTED DATA) ===\n"
                            + "\n\n".join(tool_evidence)
                        )
                    )
                    tool_evidence.clear()
                clean.append(AIMessage(content=str(msg.content)))
        elif isinstance(msg, ToolMessage):
            tool_name = str(getattr(msg, "name", "") or "tool")
            body = str(getattr(msg, "content", "") or "").strip()
            if body:
                tool_evidence.append(f"[Tool: {tool_name}]\n{body}")

    if tool_evidence:
        clean.append(
            SystemMessage(
                content="=== OBSERVED TOOL OUTPUTS (UNTRUSTED DATA) ===\n"
                + "\n\n".join(tool_evidence)
            )
        )
    return clean


def _executed_call_signatures_in_turn(messages: Sequence[BaseMessage]) -> set[tuple[str, str]]:
    """Collect canonical (name, args_json) pairs for tool calls already executed in the latest user turn."""
    latest_human_idx = -1
    for idx, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            latest_human_idx = idx
    turn_slice = messages[latest_human_idx + 1 :] if latest_human_idx >= 0 else list(messages)
    executed_ids = {
        str(getattr(msg, "tool_call_id", "") or "")
        for msg in turn_slice
        if isinstance(msg, ToolMessage)
    }
    sigs: set[tuple[str, str]] = set()
    for msg in turn_slice:
        if not isinstance(msg, AIMessage):
            continue
        for tc in getattr(msg, "tool_calls", None) or []:
            tc_id = str(tc.get("id") or "")
            if tc_id and tc_id not in executed_ids:
                continue
            name = str(tc.get("name") or "")
            args = dict(tc.get("args") or {})
            sigs.add((name, json.dumps(args, sort_keys=True, default=str)))
    return sigs


def _normalize_agent_tool_calls(
    raw_calls: Sequence[Dict[str, Any]],
    valid_tool_names: set[str],
) -> List[Dict[str, Any]]:
    """Normalize tool calls and remap accidental Harmony browser.* names onto registered tools."""
    normalized: List[Dict[str, Any]] = []
    for tc in raw_calls:
        name = str(tc.get("name") or "").strip()
        args = dict(tc.get("args") or {})
        cid = str(tc.get("id") or f"call_{uuid.uuid4().hex[:8]}")

        if name in {"browser.open", "browser_open", "open_search_result"} and "open_web_page" in valid_tool_names:
            remapped: Dict[str, Any] = {}
            if args.get("url"):
                remapped["url"] = str(args["url"])
            if "id" in args and args["id"] is not None:
                remapped["result_id"] = int(args["id"]) + 1 if int(args["id"]) == 0 else int(args["id"])
            elif args.get("result_id") is not None:
                remapped["result_id"] = int(args["result_id"])
            if args.get("cursor") is not None:
                remapped["cursor"] = max(0, int(args["cursor"]))
            name = "open_web_page"
            args = remapped
        elif name in {"browser.search", "browser_search"} and "web_search" in valid_tool_names:
            name = "web_search"
            args = {"query": str(args.get("query") or args.get("q") or "").strip()}

        if not name or (valid_tool_names and name not in valid_tool_names):
            logger.warning("Dropping unregistered LLM tool call '%s'.", name)
            continue
        normalized.append({"id": cid, "name": name, "args": args})
    return normalized


def _collect_web_sources_from_messages(
    messages: Sequence[BaseMessage],
    existing_sources: Sequence[Dict[str, str]] = (),
) -> List[Dict[str, str]]:
    """Collect structured web search + opened webpage citations from the current turn's ToolMessages."""
    latest_human_idx = -1
    for idx, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            latest_human_idx = idx
    turn_slice = messages[latest_human_idx + 1 :] if latest_human_idx >= 0 else list(messages)

    by_url: Dict[str, Dict[str, str]] = {}
    ordered_urls: List[str] = []

    def _upsert(entry: Dict[str, str]) -> None:
        url = str(entry.get("url") or "").strip()
        if not url:
            return
        if url not in by_url:
            ordered_urls.append(url)
            by_url[url] = dict(entry)
        else:
            prev = by_url[url]
            if entry.get("status") == "read" or not prev.get("snippet"):
                prev.update({k: v for k, v in entry.items() if v})

    for item in existing_sources:
        _upsert(dict(item))

    for msg in turn_slice:
        if not isinstance(msg, ToolMessage):
            continue
        tname = str(getattr(msg, "name", "") or "")
        body = str(getattr(msg, "content", "") or "")
        if tname == "web_search":
            for hit in parse_web_search_tool_text(body):
                _upsert(hit)
        elif tname == "open_web_page":
            url_match = re.search(r"^URL:\s*(\S+)", body, flags=re.MULTILINE)
            title_match = re.search(r"^Title:\s*(.+)$", body, flags=re.MULTILINE)
            if url_match:
                page_url = url_match.group(1).strip()
                is_err = body.startswith(("Unable to", "Blocked unsafe", "Provide either"))
                excerpt = ""
                if "\n\n" in body:
                    excerpt = body.split("\n\n", 1)[1][:260].replace("\n", " ").strip()
                _upsert(
                    {
                        "title": title_match.group(1).strip() if title_match else page_url,
                        "url": page_url,
                        "snippet": excerpt,
                        "status": "inaccessible" if is_err else "read",
                        "error": body[:200] if is_err else "",
                    }
                )

    return [by_url[u] for u in ordered_urls]


class BaseNodeSuite:
    """Base class for node suite holding shared state, LLM initialization, and helper methods."""

    def __init__(
        self,
        retriever: DocumentRetriever,
        grader: DocumentGrader,
        web_searcher: DuckDuckGoWebSearcher,
        tools: Sequence[BaseTool],
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        api_key: Optional[str] = None,
        settings: Optional[AppSettings] = None,
        mcp_manager: Optional[MCPClientManager] = None,
        approval_manager: Optional[ApprovalManager] = None,
        audit_repo: Optional[AuditLogRepository] = None,
        page_reader: Optional[SafeWebPageReader] = None,
        crag_pipeline: Optional[CRAGPipeline] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retriever = retriever
        self.grader = grader
        self.web_searcher = web_searcher
        self.page_reader = page_reader or SafeWebPageReader()
        self.crag_pipeline = crag_pipeline or CRAGPipeline(settings=self.settings)
        self.crag_pipeline.evaluator._make_llm = self._make_llm  # type: ignore[assignment]
        self.crag_pipeline.query_rewriter._make_llm = self._make_llm  # type: ignore[assignment]
        self.tools = list(tools)
        self.builtin_tools_map: Dict[str, BaseTool] = {t.name: t for t in tools}
        self.model_name = model_name or self.settings.default_text_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = (api_key if api_key is not None else self.settings.groq_api_key).strip()
        self.mcp_manager = mcp_manager
        self.approval_manager = approval_manager or (
            ApprovalManager(db_manager=mcp_manager.db) if mcp_manager is not None else None
        )
        self.audit_repo = audit_repo or (
            AuditLogRepository(mcp_manager.db) if mcp_manager is not None else None
        )

    def _make_llm(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatGroq:
        import qna_bot.agent.nodes as nodes_pkg
        chat_cls = getattr(nodes_pkg, "ChatGroq", ChatGroq)
        return chat_cls(
            model=self.model_name,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            api_key=self.api_key,
        )

    def _make_agent_llm(self, tools: Sequence[BaseTool]) -> Any:
        """Build Agent LLM with explicit tool_choice='auto' for function/MCP tool selection."""
        llm = self._make_llm()
        tool_list = list(tools)
        if not tool_list:
            return llm
        try:
            return llm.bind_tools(tool_list, tool_choice="auto")
        except TypeError:
            return llm.bind_tools(tool_list)

    def _make_response_llm(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatGroq:
        """Build Response LLM strictly without bound tools for final text synthesis."""
        return self._make_llm(temperature=temperature, max_tokens=max_tokens)

    def _invoke_response_llm(self, messages: Sequence[BaseMessage]) -> str:
        """Invoke unbound Response LLM safely, retrying once on Harmony tool_use_failed errors."""
        prompt_msgs = list(messages)
        try:
            response = self._make_response_llm().invoke(prompt_msgs)
            return str(getattr(response, "content", "") or "").strip()
        except Exception as exc:
            if not _is_tool_use_failed_error(exc):
                raise
            logger.warning("Response LLM attempted tool call despite tool_choice='none'; retrying plain synthesis: %s", exc)
            retry_msgs = [
                SystemMessage(
                    content=(
                        "STRICT PLAIN-TEXT RESPONSE ONLY: Tools are disabled. "
                        "NEVER invoke or emit browser.open, browser.search, JSON tool payloads, or function calls. "
                        "Write the complete user-facing Markdown response directly using the provided text."
                    )
                ),
                *prompt_msgs,
            ]
            response = self._make_response_llm().invoke(retry_msgs)
            return str(getattr(response, "content", "") or "").strip()

    def _invoke_agent_llm(
        self,
        lc_messages: List[BaseMessage],
        tools: Sequence[BaseTool],
        allow_text_fallback: bool = True,
    ) -> AIMessage:
        """Invoke Agent LLM with tool_choice='auto', recovering cleanly if Groq rejects Harmony tool output."""
        tool_list = list(tools)
        valid_names = ", ".join(t.name for t in tool_list[:40]) or "none"
        try:
            if tool_list:
                return self._make_agent_llm(tool_list).invoke(lc_messages)
            return self._make_response_llm().invoke(lc_messages)
        except Exception as exc:
            if not _is_tool_use_failed_error(exc):
                raise
            logger.warning("Agent LLM hit tool_use_failed (%s); retrying with registered tool schema reminder.", exc)
            reminder = SystemMessage(
                content=(
                    "Do NOT emit `browser.open`, `browser.search`, or any unregistered function name. "
                    f"Only call one of these exact registered tools: {valid_names}. "
                    "Or respond directly in plain Markdown without tool calls."
                )
            )
            try:
                if tool_list:
                    return self._make_agent_llm(tool_list).invoke([reminder, *lc_messages])
                return self._make_response_llm().invoke([reminder, *lc_messages])
            except Exception as inner_exc:
                if not (allow_text_fallback and _is_tool_use_failed_error(inner_exc)):
                    raise
                logger.warning("Agent tool retry failed (%s); falling back to Response LLM synthesis.", inner_exc)
                clean_msgs = _messages_for_response_llm(lc_messages)
                text = self._invoke_response_llm(
                    [SystemMessage(content=QNAGPT_SYSTEM_PROMPT + RESPONSE_ONLY_SYSTEM_ADDENDUM), *clean_msgs]
                )
                return AIMessage(content=text)

    def _list_user_docs_safe(self, user_id: str) -> List[Dict[str, Any]]:
        try:
            return self.retriever.vector_store.list_documents(user_id=user_id)
        except Exception as exc:
            logger.warning("Could not list indexed documents during analysis: %s", exc)
            return []

    def _list_mcp_tools_safe(self, user_id: str) -> List[DiscoveredTool]:
        if self.mcp_manager is None:
            return []
        try:
            return [
                t
                for t in self.mcp_manager.list_available_tools(user_id=user_id)
                if t.permission_category != "BLOCKED"
            ]
        except Exception as exc:
            logger.warning("Could not list MCP tools during analysis: %s", exc)
            return []

    @staticmethod
    def _turn_reset_fields(
        *,
        query_type: str,
        doc_required: bool,
        web_required: bool,
        activity: str,
        max_iterations: int = 10,
    ) -> Dict[str, Any]:
        return {
            "query_type": query_type,
            "document_search_required": doc_required,
            "web_search_required": web_required,
            "activity": activity,
            "rewritten_query": "",
            "answer": "",
            "validation_passed": True,
            "retry_count": 0,
            "validation_retries": 0,
            "tool_calls_count": 0,
            "retrieved_documents": [],
            "relevant_documents": [],
            "web_results": [],
            "opened_webpages": [],
            "verified_research": {},
            "sources": [],
            "web_sources": [],
            "context": "",
            "crag_enabled": True,
            "crag_status": "PENDING",
            "crag_decision": "",
            "document_evaluations": [],
            "evidence_bundle": {},
            "retrieval_attempt_count": 0,
            "contradiction_detected": False,
            "web_retry_count": 0,
            "available_tools": [],
            "planned_steps": [],
            "pending_tool_calls": [],
            "pending_approval": None,
            "approval_decision": None,
            "tool_executions": [],
            "iteration_count": 0,
            "max_iterations": max_iterations,
            "agent_status": "Planning",
            "needs_more_steps": False,
            "error": "",
        }
