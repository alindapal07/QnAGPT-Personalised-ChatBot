"""Service layer for user-scoped conversation threads, ORM history, grouping, archiving, and feedback."""

from collections import OrderedDict
from typing import Any, Dict, List, Optional

from qna_bot.agent.memory import ConversationMemoryManager
from qna_bot.database.session import SQLAlchemyDatabase
from qna_bot.repositories import ConversationRepository

_BUCKET_ORDER = ("Pinned", "Today", "Yesterday", "Previous 7 Days", "Previous 30 Days", "Older")


class ConversationService:
    """High-level conversation management coordinating SQLAlchemy ORM + LangGraph memory tables."""

    def __init__(
        self,
        memory_manager: Optional[ConversationMemoryManager] = None,
        db: Optional[SQLAlchemyDatabase] = None,
    ) -> None:
        self.memory = memory_manager or ConversationMemoryManager()
        self.db = db
        self.repo = ConversationRepository(db) if db is not None else None

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
        if self.repo is not None:
            try:
                self.repo.create(
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

    def list_threads_grouped(
        self,
        user_id: str,
        search_query: str = "",
    ) -> OrderedDict[str, List[Dict[str, Any]]]:
        archived_ids: set[str] = set()
        if self.repo is not None:
            try:
                active_ids = {c.id for c in self.repo.list_for_user(user_id=user_id, include_archived=False)}
                all_orm = self.repo.list_for_user(user_id=user_id, include_archived=True)
                archived_ids = {c.id for c in all_orm if c.id not in active_ids}
            except Exception:
                archived_ids = set()

        threads = [
            t
            for t in self.memory.retrieve_all_threads(user_id=user_id, search_query=search_query)
            if t.get("message_count", 0) > 0 and str(t.get("thread_id", "")) not in archived_ids
        ]
        grouped: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()
        for bucket in _BUCKET_ORDER:
            grouped[bucket] = []

        for t in threads:
            if t.get("is_pinned"):
                grouped["Pinned"].append(t)
            else:
                bucket = str(t.get("bucket") or "Older")
                grouped.setdefault(bucket, []).append(t)

        return OrderedDict((k, v) for k, v in grouped.items() if v)

    def load_history(self, thread_id: str, user_id: str) -> List[Dict[str, Any]]:
        return self.memory.retrieve_chat_history(thread_id=thread_id, user_id=user_id)

    def rename_thread(self, thread_id: str, user_id: str, new_title: str) -> bool:
        ok = self.memory.rename_thread(thread_id=thread_id, user_id=user_id, new_title=new_title)
        if ok and self.repo is not None:
            try:
                self.repo.rename(thread_id, user_id, new_title)
            except Exception:
                pass
        return ok

    def set_pinned(self, thread_id: str, user_id: str, pinned: bool) -> bool:
        ok = self.memory.set_thread_pinned(thread_id=thread_id, user_id=user_id, pinned=pinned)
        if ok and self.repo is not None:
            try:
                self.repo.set_pinned(thread_id, user_id, pinned)
            except Exception:
                pass
        return ok

    def set_archived(self, thread_id: str, user_id: str, archived: bool) -> bool:
        if self.repo is None:
            return False
        return self.repo.set_archived(thread_id, user_id, archived)

    def delete_thread(self, thread_id: str, user_id: str) -> bool:
        ok = self.memory.delete_thread(thread_id=thread_id, user_id=user_id)
        if self.repo is not None:
            try:
                self.repo.delete(thread_id, user_id)
            except Exception:
                pass
        return ok

    def list_executions(self, user_id: str, limit: int = 25) -> List[Dict[str, Any]]:
        if self.repo is None:
            return []
        return [
            {
                "execution_id": ex.execution_id,
                "conversation_id": ex.conversation_id or "",
                "status": ex.status,
                "current_step": ex.current_step,
                "started_at": ex.started_at.isoformat() if ex.started_at else "",
                "completed_at": ex.completed_at.isoformat() if ex.completed_at else "",
            }
            for ex in self.repo.list_executions(user_id=user_id, limit=limit)
        ]

    def record_feedback(
        self,
        user_id: str,
        conversation_id: str,
        rating: str,
        comment: str = "",
        message_content: str = "",
    ) -> bool:
        if self.repo is None:
            return False
        msgs = self.repo.list_messages(conversation_id, user_id)
        if not msgs:
            return False
        target = None
        for m in reversed(msgs):
            if m.role == "assistant" and (not message_content or m.content == message_content):
                target = m
                break
        if target is None:
            target = next((m for m in reversed(msgs) if m.role == "assistant"), msgs[-1])
        self.repo.record_feedback(user_id=user_id, message_id=target.id, rating=rating, comment=comment)
        return True
