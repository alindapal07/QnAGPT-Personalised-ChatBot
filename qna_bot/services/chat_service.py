"""Chat orchestration service combining LangGraph Agentic RAG, Vision fallback, and SQLite memory."""

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Dict, Generator, List, Mapping, Optional, Sequence

from langchain_core.messages import HumanMessage

from qna_bot.agent.graph import build_agent_graph
from qna_bot.agent.memory import ConversationMemoryManager
from qna_bot.config.settings import AppSettings, get_settings
from qna_bot.core.logging_config import get_logger
from qna_bot.rag.grader import DocumentGrader
from qna_bot.rag.retriever import DocumentRetriever
from qna_bot.rag.vector_store import FAISSVectorStoreManager
from qna_bot.services.agent_service import AIAgent
from qna_bot.tools.web_search import DuckDuckGoWebSearcher

logger = get_logger(__name__)


@dataclass(frozen=True)
class ChatTurnResult:
    """Final structured output of a single chat turn."""

    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    web_sources: List[Dict[str, str]] = field(default_factory=list)
    query_type: str = "GENERAL"
    error: str = ""


class ChatService:
    """Coordinates LangGraph streaming execution, multimodal vision routing, and SQLite threads."""

    def __init__(
        self,
        settings: Optional[AppSettings] = None,
        memory_manager: Optional[ConversationMemoryManager] = None,
        vector_store: Optional[FAISSVectorStoreManager] = None,
        web_searcher: Optional[DuckDuckGoWebSearcher] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.memory = memory_manager or ConversationMemoryManager(
            db_path=self.settings.sqlite_db_path
        )
        self.vector_store = vector_store or FAISSVectorStoreManager(
            base_dir=self.settings.vector_store_dir
        )
        self.web_searcher = web_searcher or DuckDuckGoWebSearcher()

    def create_thread(
        self,
        user_id: str,
        title: str = "New Conversation",
        chat_mode: str = "Auto (Agentic)",
        thread_id: Optional[str] = None,
    ) -> str:
        return self.memory.create_new_thread(
            user_id=user_id,
            title=title,
            chat_mode=chat_mode,
            thread_id=thread_id,
        )

    def list_threads(self, user_id: str) -> List[Dict[str, Any]]:
        return self.memory.retrieve_all_threads(user_id=user_id)

    def load_thread_history(self, thread_id: str, user_id: str) -> List[Dict[str, Any]]:
        return self.memory.retrieve_chat_history(thread_id=thread_id, user_id=user_id)

    def delete_thread(self, thread_id: str, user_id: str) -> bool:
        return self.memory.delete_thread(thread_id=thread_id, user_id=user_id)

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
    ) -> ChatTurnResult:
        """Execute the compiled LangGraph Agentic RAG graph with live node activity updates."""
        self.memory.ensure_thread(
            thread_id=thread_id,
            user_id=user_id,
            chat_mode=chat_mode,
        )
        retriever = DocumentRetriever(vector_store=self.vector_store, settings=self.settings)
        grader = DocumentGrader(
            model_name=model_name,
            api_key=self.settings.groq_api_key,
            settings=self.settings,
        )
        graph = build_agent_graph(
            user_id=user_id,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=self.settings.groq_api_key,
            retriever=retriever,
            grader=grader,
            web_searcher=self.web_searcher,
            memory_manager=self.memory,
            settings=self.settings,
        )

        input_state = {
            "messages": [HumanMessage(content=user_query)],
            "user_query": user_query,
            "chat_mode": chat_mode,
            "user_id": user_id,
            "thread_id": thread_id,
            "system_prompt": system_prompt,
        }
        config = {"configurable": {"thread_id": thread_id}}

        final_state: Dict[str, Any] = {}
        for event in graph.stream(input_state, config=config, stream_mode="updates"):
            for node_name, update in event.items():
                if not isinstance(update, dict):
                    continue
                final_state.update(update)
                activity = update.get("activity")
                if activity and on_activity:
                    on_activity(str(activity))
                logger.debug("Graph node '%s' completed (thread=%s)", node_name, thread_id)

        return ChatTurnResult(
            answer=str(final_state.get("answer") or ""),
            sources=list(final_state.get("sources") or []),
            web_sources=list(final_state.get("web_sources") or []),
            query_type=str(final_state.get("query_type") or "GENERAL"),
            error=str(final_state.get("error") or ""),
        )

    def stream_text_tokens(self, text: str) -> Generator[str, None, None]:
        """Stream a completed response smoothly by word tokens for Streamlit write_stream."""
        delay = self.settings.stream_chunk_delay_sec
        tokens = text.split(" ")
        for idx, word in enumerate(tokens):
            if delay > 0:
                time.sleep(min(delay, 0.015))
            yield word if idx == len(tokens) - 1 else f"{word} "
