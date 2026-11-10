"""Context-aware dynamic prompt suggestion service with signature caching and graceful fallback."""

from dataclasses import dataclass
import hashlib
import re
from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Tuple

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from qna_bot.config.settings import AppSettings, get_settings
from qna_bot.core.logging_config import get_logger
from qna_bot.services.conversation_service import ConversationService
from qna_bot.services.document_service import DocumentService
from qna_bot.services.mcp_service import MCPService

logger = get_logger(__name__)


@dataclass(frozen=True)
class SuggestedPrompt:
    """Compact contextual suggestion rendered on an empty Chat canvas."""

    label: str
    prompt: str
    category: str
    icon: str = ":material/search:"


class _SuggestionItemSchema(BaseModel):
    label: str = Field(..., min_length=4, max_length=48)
    prompt: str = Field(..., min_length=10, max_length=160)
    category: str = Field(default="Workspace")


class _SuggestionBatchSchema(BaseModel):
    suggestions: List[_SuggestionItemSchema] = Field(default_factory=list)


_CATEGORY_ICON_MAP: Dict[str, str] = {
    "Research": ":material/search:",
    "Documents": ":material/library_books:",
    "MCP": ":material/extension:",
    "Follow-up": ":material/chat:",
    "Build": ":material/build:",
    "Workspace": ":material/smart_toy:",
}


def _dedupe_suggestions(items: Sequence[SuggestedPrompt], limit: int = 4) -> List[SuggestedPrompt]:
    seen_prompts: set[str] = set()
    seen_labels: set[str] = set()
    out: List[SuggestedPrompt] = []
    for item in items:
        norm_p = re.sub(r"\s+", " ", (item.prompt or "").strip()).lower()
        norm_l = re.sub(r"\s+", " ", (item.label or "").strip()).lower()
        if not norm_p or not norm_l or norm_p in seen_prompts or norm_l in seen_labels:
            continue
        seen_prompts.add(norm_p)
        seen_labels.add(norm_l)
        out.append(item)
        if len(out) >= limit:
            break
    return out


class SuggestionService:
    """Generates and caches relevant Chat starter prompts from live workspace context."""

    def __init__(
        self,
        settings: Optional[AppSettings] = None,
        mcp_service: Optional[MCPService] = None,
        document_service: Optional[DocumentService] = None,
        conversation_service: Optional[ConversationService] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.mcp_service = mcp_service
        self.document_service = document_service
        self.conversation_service = conversation_service

    def collect_context(
        self,
        user_id: str,
        current_chat_id: str = "",
        model_name: str = "",
    ) -> Dict[str, Any]:
        """Selectively collect relevant MCP, Knowledge Base, and recent thread metadata."""
        connected_servers: List[Dict[str, Any]] = []
        tool_highlights: List[Dict[str, str]] = []
        if self.mcp_service is not None:
            try:
                statuses = self.mcp_service.list_statuses(user_id=user_id)
                connected_servers = [
                    {"name": s.name, "tool_count": s.tool_count, "transport": s.transport}
                    for s in statuses
                    if s.connected
                ]
                enabled_tools = [
                    t
                    for t in self.mcp_service.list_tools(user_id=user_id)
                    if t.enabled and t.permission_category != "BLOCKED"
                ]
                for t in enabled_tools[:30]:
                    tool_highlights.append(
                        {
                            "server": t.server_name,
                            "name": t.original_name,
                            "description": (t.description or "")[:90],
                        }
                    )
            except Exception as exc:
                logger.debug("Suggestion context skipped MCP inspection: %s", exc)

        docs: List[Dict[str, Any]] = []
        if self.document_service is not None:
            try:
                for d in self.document_service.list_documents(user_id=user_id)[:8]:
                    docs.append(
                        {
                            "filename": str(d.get("filename") or "Document"),
                            "file_type": str(d.get("file_type") or ""),
                            "chunk_count": int(d.get("chunk_count") or 0),
                            "page_count": d.get("page_count"),
                        }
                    )
            except Exception as exc:
                logger.debug("Suggestion context skipped docs inspection: %s", exc)

        recent_topics: List[str] = []
        if self.conversation_service is not None:
            try:
                threads = self.conversation_service.memory.retrieve_all_threads(user_id=user_id)
                for th in threads:
                    tid = str(th.get("thread_id") or "")
                    title = str(th.get("title") or "").strip()
                    if tid == current_chat_id:
                        continue
                    if title and title not in {"New Conversation", "Conversation"} and int(th.get("message_count") or 0) > 0:
                        recent_topics.append(title)
                    if len(recent_topics) >= 5:
                        break
            except Exception as exc:
                logger.debug("Suggestion context skipped thread history: %s", exc)

        raw_sig = "|".join(
            [
                user_id,
                current_chat_id,
                model_name or self.settings.default_text_model,
                ",".join(f"{s['name']}:{s['tool_count']}" for s in connected_servers),
                ",".join(f"{t['server']}:{t['name']}" for t in tool_highlights[:12]),
                ",".join(f"{d['filename']}:{d['chunk_count']}" for d in docs[:6]),
                ",".join(recent_topics[:4]),
            ]
        )
        sig_hash = hashlib.sha256(raw_sig.encode("utf-8")).hexdigest()[:16]

        return {
            "signature": sig_hash,
            "connected_servers": connected_servers,
            "tool_highlights": tool_highlights,
            "documents": docs,
            "recent_topics": recent_topics,
            "model_name": model_name or self.settings.default_text_model,
        }

    def _build_deterministic_suggestions(self, ctx: Dict[str, Any]) -> List[SuggestedPrompt]:
        """Generate grounded, capability-specific suggestions without external calls."""
        candidates: List[SuggestedPrompt] = []
        servers = ctx.get("connected_servers") or []
        tools = ctx.get("tool_highlights") or []
        docs = ctx.get("documents") or []
        topics = ctx.get("recent_topics") or []

        # 1. MCP Server / Discovered Tool-aware prompts (never assume unexposed capabilities)
        if tools:
            tool_blob = " ".join(
                f"{t['server']} {t['name']} {t['description']}".lower() for t in tools
            )
            srv_names = [s["name"] for s in servers]
            primary_srv = srv_names[0] if srv_names else tools[0]["server"]

            if any(k in tool_blob for k in ("startup", "theresanaiforthat", "company", "founder", "crunchbase", "funding", "business")):
                candidates.append(
                    SuggestedPrompt(
                        label="Research startups via MCP & web",
                        prompt=f"Find five AI startups in India using {primary_srv} and verify their founders, websites, and funding through web search.",
                        category="MCP",
                        icon=_CATEGORY_ICON_MAP["MCP"],
                    )
                )
            elif any(k in tool_blob for k in ("linkedin", "profile", "person", "job", "company_profile")):
                candidates.append(
                    SuggestedPrompt(
                        label=f"Query {primary_srv} profiles",
                        prompt=f"Use {primary_srv} tools to look up company or founder details and summarize verified highlights.",
                        category="MCP",
                        icon=_CATEGORY_ICON_MAP["MCP"],
                    )
                )
            elif any(k in tool_blob for k in ("repo", "github", "pull_request", "issue", "commit")):
                candidates.append(
                    SuggestedPrompt(
                        label="Inspect repository with MCP",
                        prompt=f"Search repositories or open issues via {primary_srv} and summarize key engineering updates.",
                        category="MCP",
                        icon=_CATEGORY_ICON_MAP["MCP"],
                    )
                )
            elif any(k in tool_blob for k in ("read_file", "filesystem", "directory", "sql", "database", "query")):
                candidates.append(
                    SuggestedPrompt(
                        label=f"Run read-only {primary_srv} task",
                        prompt=f"Inspect available workspace data with {primary_srv} ({tools[0]['name']}) and present a structured summary.",
                        category="MCP",
                        icon=_CATEGORY_ICON_MAP["MCP"],
                    )
                )
            else:
                sample_tool = tools[0]["name"]
                candidates.append(
                    SuggestedPrompt(
                        label=f"Explore {primary_srv} capabilities",
                        prompt=f"Use {primary_srv} (`{sample_tool}`) to complete a read-only lookup and summarize the results.",
                        category="MCP",
                        icon=_CATEGORY_ICON_MAP["MCP"],
                    )
                )

        # 2. Knowledge Base Document-aware prompts
        if len(docs) >= 2:
            d1, d2 = docs[0]["filename"], docs[1]["filename"]
            candidates.append(
                SuggestedPrompt(
                    label="Compare uploaded documents",
                    prompt=f"Compare key findings, methodology, and differences between '{d1}' and '{d2}' with page citations.",
                    category="Documents",
                    icon=_CATEGORY_ICON_MAP["Documents"],
                )
            )
        elif len(docs) == 1:
            d1 = docs[0]["filename"]
            candidates.append(
                SuggestedPrompt(
                    label=f"Summarize {d1[:26]}",
                    prompt=f"Summarize the core findings, structure, and action items in '{d1}' with source citations.",
                    category="Documents",
                    icon=_CATEGORY_ICON_MAP["Documents"],
                )
            )

        # 3. Recent conversation context follow-up (avoid repeating verbatim)
        if topics:
            latest_topic = re.sub(r"\.\.\.$", "", topics[0]).strip()
            if latest_topic:
                short_lbl = latest_topic[:26] + ("..." if len(latest_topic) > 26 else "")
                candidates.append(
                    SuggestedPrompt(
                        label=f"Follow up: {short_lbl}",
                        prompt=f"Provide a structured trade-off comparison and implementation checklist related to: {latest_topic}",
                        category="Follow-up",
                        icon=_CATEGORY_ICON_MAP["Follow-up"],
                    )
                )

        # 4. Capability-aligned baseline suggestions (non-repetitive fallbacks)
        fallbacks = (
            SuggestedPrompt(
                label="Research a market topic with citations",
                prompt="Research five emerging enterprise AI infrastructure companies, verify their websites across sources, and compare them in a table.",
                category="Research",
                icon=_CATEGORY_ICON_MAP["Research"],
            ),
            SuggestedPrompt(
                label="Compare technical architectures",
                prompt="Compare FastAPI, LangGraph, and SQLite checkpointing patterns for production stateful AI backends.",
                category="Build",
                icon=_CATEGORY_ICON_MAP["Build"],
            ),
            SuggestedPrompt(
                label="Verify sources across webpages",
                prompt="Search the web for the latest Python 3.13 performance features, open the official release notes, and summarize key highlights.",
                category="Research",
                icon=_CATEGORY_ICON_MAP["Research"],
            ),
            SuggestedPrompt(
                label="Review connected tools & safety",
                prompt="Review which workspace tools and document collections are currently available and recommend a safe workflow.",
                category="Workspace",
                icon=_CATEGORY_ICON_MAP["Workspace"],
            ),
        )
        candidates.extend(fallbacks)
        return _dedupe_suggestions(candidates, limit=4)

    def _refine_with_llm(
        self,
        ctx: Dict[str, Any],
        model_name: str,
    ) -> List[SuggestedPrompt]:
        """Use Groq concisely only when live context warrants fresh phrasing."""
        api_key = (self.settings.groq_api_key or "").strip()
        if not api_key:
            return []

        srv_str = ", ".join(f"{s['name']} ({s['tool_count']} tools)" for s in ctx["connected_servers"]) or "None"
        tool_str = ", ".join(f"{t['server']}.{t['name']}" for t in ctx["tool_highlights"][:12]) or "None"
        doc_str = ", ".join(d["filename"] for d in ctx["documents"][:5]) or "None"
        hist_str = " | ".join(ctx["recent_topics"][:3]) or "None"

        prompt = (
            "Generate 4 concise, practical starter prompts for a user opening a new QnAGPT workspace chat.\n"
            "Strictly ground every suggestion ONLY in the listed tools, documents, or recent topics. "
            "Never reference tools or documents that are 'None'. Keep labels 3-6 words and prompts 1 concise sentence.\n\n"
            f"Connected MCP Servers: {srv_str}\n"
            f"Sample Discovered Tools: {tool_str}\n"
            f"Uploaded Knowledge Base Files: {doc_str}\n"
            f"Recent Conversation Topics (avoid repeating verbatim): {hist_str}"
        )
        llm = ChatGroq(
            model=model_name or self.settings.default_text_model,
            temperature=0.3,
            max_tokens=320,
            api_key=api_key,
            max_retries=0,
            request_timeout=4.0,
        )
        parsed = llm.with_structured_output(_SuggestionBatchSchema).invoke(prompt)
        out: List[SuggestedPrompt] = []
        for item in (parsed.suggestions or [])[:4]:
            cat = item.category if item.category in _CATEGORY_ICON_MAP else "Workspace"
            out.append(
                SuggestedPrompt(
                    label=item.label.strip(),
                    prompt=item.prompt.strip(),
                    category=cat,
                    icon=_CATEGORY_ICON_MAP.get(cat, ":material/search:"),
                )
            )
        return _dedupe_suggestions(out, limit=4)

    def get_suggestions(
        self,
        user_id: str,
        current_chat_id: str = "",
        model_name: str = "",
        cache_store: Optional[MutableMapping[str, Any]] = None,
        force_refresh: bool = False,
        allow_llm: bool = False,
    ) -> List[SuggestedPrompt]:
        """Return 4 relevant cached suggestions, refreshing only when workspace context changes."""
        ctx = self.collect_context(
            user_id=user_id,
            current_chat_id=current_chat_id,
            model_name=model_name,
        )
        sig = ctx["signature"]

        if cache_store is not None and not force_refresh:
            cached = cache_store.get("_qnagpt_suggestions_cache")
            if isinstance(cached, dict) and cached.get("signature") == sig and cached.get("items"):
                return list(cached["items"])

        suggestions = self._build_deterministic_suggestions(ctx)
        has_rich_context = bool(ctx["tool_highlights"] or ctx["documents"] or ctx["recent_topics"])
        if allow_llm and has_rich_context and (self.settings.groq_api_key or "").strip():
            try:
                llm_items = self._refine_with_llm(ctx, model_name=ctx["model_name"])
                if len(llm_items) >= 3:
                    suggestions = _dedupe_suggestions([*llm_items, *suggestions], limit=4)
            except Exception as exc:
                logger.debug("LLM suggestion refinement fell back deterministically: %s", exc)

        if cache_store is not None:
            cache_store["_qnagpt_suggestions_cache"] = {
                "signature": sig,
                "items": list(suggestions),
            }
        return suggestions
