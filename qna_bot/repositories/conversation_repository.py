"""User-scoped SQLAlchemy repositories for conversations, messages, executions, feedback, and audit logs."""

from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Mapping, Optional
import uuid

from sqlalchemy import delete, select

from qna_bot.core.exceptions import AuthorizationError
from qna_bot.database.session import SQLAlchemyDatabase
from qna_bot.models.audit import AuditLog
from qna_bot.models.conversation import AgentExecution, Conversation, FeedbackRecord, Message
from qna_bot.models.mcp import MCPExecutionLog


class ConversationRepository:
    """Strictly user-scoped persistence for conversations, messages, runs, and feedback."""

    def __init__(self, db: SQLAlchemyDatabase) -> None:
        self.db = db

    def create(
        self,
        user_id: str,
        title: str = "New Conversation",
        agent_mode: str = "Auto (Agentic)",
        model_name: str = "openai/gpt-oss-20b",
        conversation_id: Optional[str] = None,
    ) -> Conversation:
        cid = conversation_id or f"conversation_{uuid.uuid4().hex[:12]}"
        with self.db.session_scope() as session:
            existing = session.get(Conversation, cid)
            if existing is not None:
                if existing.user_id != user_id:
                    raise AuthorizationError("Conversation ID belongs to another user.")
                return existing
            conv = Conversation(
                id=cid,
                user_id=user_id,
                title=title.strip() or "New Conversation",
                agent_mode=agent_mode,
                model_name=model_name,
            )
            session.add(conv)
            session.flush()
            session.refresh(conv)
            return conv

    def get(self, conversation_id: str, user_id: str) -> Optional[Conversation]:
        with self.db.session_scope() as session:
            return session.scalar(
                select(Conversation).where(
                    Conversation.id == conversation_id,
                    Conversation.user_id == user_id,
                )
            )

    def list_for_user(
        self,
        user_id: str,
        search_query: str = "",
        include_archived: bool = False,
    ) -> List[Conversation]:
        query = (search_query or "").strip().lower()
        with self.db.session_scope() as session:
            stmt = select(Conversation).where(Conversation.user_id == user_id)
            if not include_archived:
                stmt = stmt.where(Conversation.status != "archived")
            stmt = stmt.order_by(Conversation.is_pinned.desc(), Conversation.updated_at.desc())
            convs = list(session.scalars(stmt).all())
            if not query:
                return convs

            matched_conv_ids = set(
                session.scalars(
                    select(Message.conversation_id)
                    .where(Message.user_id == user_id, Message.content.ilike(f"%{query}%"))
                ).all()
            )
            return [
                c
                for c in convs
                if query in c.title.lower()
                or query in (c.description or "").lower()
                or c.id in matched_conv_ids
            ]

    def rename(self, conversation_id: str, user_id: str, new_title: str) -> bool:
        cleaned = (new_title or "").strip()[:255]
        if not cleaned:
            return False
        with self.db.session_scope() as session:
            conv = session.scalar(
                select(Conversation).where(
                    Conversation.id == conversation_id,
                    Conversation.user_id == user_id,
                )
            )
            if conv is None:
                return False
            conv.title = cleaned
            conv.is_custom_title = True
            conv.updated_at = datetime.now(timezone.utc)
            return True

    def set_pinned(self, conversation_id: str, user_id: str, pinned: bool) -> bool:
        with self.db.session_scope() as session:
            conv = session.scalar(
                select(Conversation).where(
                    Conversation.id == conversation_id,
                    Conversation.user_id == user_id,
                )
            )
            if conv is None:
                return False
            conv.is_pinned = bool(pinned)
            conv.updated_at = datetime.now(timezone.utc)
            return True

    def set_archived(self, conversation_id: str, user_id: str, archived: bool) -> bool:
        now = datetime.now(timezone.utc)
        with self.db.session_scope() as session:
            conv = session.scalar(
                select(Conversation).where(
                    Conversation.id == conversation_id,
                    Conversation.user_id == user_id,
                )
            )
            if conv is None:
                return False
            conv.status = "archived" if archived else "active"
            conv.archived_at = now if archived else None
            conv.updated_at = now
            return True

    def delete(self, conversation_id: str, user_id: str) -> bool:
        with self.db.session_scope() as session:
            res = session.execute(
                delete(Conversation).where(
                    Conversation.id == conversation_id,
                    Conversation.user_id == user_id,
                )
            )
            return int(res.rowcount or 0) > 0

    def append_message(
        self,
        conversation_id: str,
        user_id: str,
        role: str,
        content: str,
        message_type: str = "text",
        token_usage: int = 0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Message:
        now = datetime.now(timezone.utc)
        with self.db.session_scope() as session:
            conv = session.get(Conversation, conversation_id)
            if conv is None:
                conv = Conversation(
                    id=conversation_id,
                    user_id=user_id,
                    title="New Conversation",
                )
                session.add(conv)
                session.flush()
            elif conv.user_id != user_id:
                raise AuthorizationError("Cannot write message into another user's conversation.")

            msg = Message(
                conversation_id=conversation_id,
                user_id=user_id,
                role=role,
                content=content,
                message_type=message_type,
                token_usage=int(token_usage or 0),
                metadata_json=json.dumps(dict(metadata or {}), default=str),
                created_at=now,
            )
            session.add(msg)
            conv.updated_at = now
            if role == "user" and not conv.is_custom_title and conv.title in ("New Conversation", "Conversation"):
                snippet = " ".join((content or "").split())
                if snippet:
                    conv.title = (snippet[:117] + "...") if len(snippet) > 120 else snippet
            session.flush()
            session.refresh(msg)
            return msg

    def list_messages(self, conversation_id: str, user_id: str) -> List[Message]:
        with self.db.session_scope() as session:
            conv = session.scalar(
                select(Conversation).where(
                    Conversation.id == conversation_id,
                    Conversation.user_id == user_id,
                )
            )
            if conv is None:
                return []
            return list(
                session.scalars(
                    select(Message)
                    .where(Message.conversation_id == conversation_id, Message.user_id == user_id)
                    .order_by(Message.id.asc())
                ).all()
            )

    def start_execution(
        self,
        user_id: str,
        conversation_id: Optional[str],
        execution_id: str,
        current_step: str = "analyze_query",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> AgentExecution:
        with self.db.session_scope() as session:
            rec = AgentExecution(
                user_id=user_id,
                conversation_id=conversation_id,
                execution_id=execution_id,
                status="running",
                current_step=current_step,
                execution_metadata=json.dumps(dict(metadata or {}), default=str),
            )
            session.add(rec)
            session.flush()
            session.refresh(rec)
            return rec

    def finish_execution(
        self,
        user_id: str,
        execution_id: str,
        status: str,
        current_step: str = "",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Optional[AgentExecution]:
        now = datetime.now(timezone.utc)
        with self.db.session_scope() as session:
            rec = session.scalar(
                select(AgentExecution).where(
                    AgentExecution.execution_id == execution_id,
                    AgentExecution.user_id == user_id,
                )
            )
            if rec is None:
                return None
            rec.status = status
            if current_step:
                rec.current_step = current_step
            if metadata is not None:
                rec.execution_metadata = json.dumps(dict(metadata), default=str)
            rec.completed_at = now
            session.flush()
            session.refresh(rec)
            return rec

    def list_executions(self, user_id: str, limit: int = 25) -> List[AgentExecution]:
        with self.db.session_scope() as session:
            return list(
                session.scalars(
                    select(AgentExecution)
                    .where(AgentExecution.user_id == user_id)
                    .order_by(AgentExecution.started_at.desc())
                    .limit(limit)
                ).all()
            )

    def record_feedback(
        self,
        user_id: str,
        message_id: int,
        rating: str,
        comment: str = "",
    ) -> FeedbackRecord:
        with self.db.session_scope() as session:
            msg = session.get(Message, message_id)
            if msg is None or msg.user_id != user_id:
                raise AuthorizationError("Cannot submit feedback on a message outside your account.")
            fb = FeedbackRecord(
                user_id=user_id,
                message_id=message_id,
                rating=rating,
                comment=comment.strip()[:2000],
            )
            session.add(fb)
            session.flush()
            session.refresh(fb)
            return fb


class AuditAndTelemetryRepository:
    """Writes immutable security audit logs and MCP execution telemetry rows."""

    def __init__(self, db: SQLAlchemyDatabase) -> None:
        self.db = db

    def log_audit(
        self,
        user_id: Optional[str],
        action: str,
        resource_type: str,
        resource_id: str = "",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> AuditLog:
        with self.db.session_scope() as session:
            entry = AuditLog(
                user_id=user_id,
                action=action[:100],
                resource_type=resource_type[:64],
                resource_id=str(resource_id)[:128],
                metadata_json=json.dumps(dict(metadata or {}), default=str),
            )
            session.add(entry)
            session.flush()
            session.refresh(entry)
            return entry

    def log_mcp_execution(
        self,
        user_id: str,
        tool_name: str,
        status: str,
        execution_time: float = 0.0,
        conversation_id: Optional[str] = None,
        server_id: Optional[str] = None,
        error_message: str = "",
    ) -> MCPExecutionLog:
        with self.db.session_scope() as session:
            row = MCPExecutionLog(
                user_id=user_id,
                conversation_id=conversation_id,
                server_id=server_id,
                tool_name=tool_name[:160],
                status=status[:32],
                execution_time=float(execution_time),
                error_message=str(error_message or "")[:2000],
            )
            session.add(row)
            session.flush()
            session.refresh(row)
            return row

    def list_audit_logs(self, user_id: str, limit: int = 50) -> List[AuditLog]:
        with self.db.session_scope() as session:
            return list(
                session.scalars(
                    select(AuditLog)
                    .where(AuditLog.user_id == user_id)
                    .order_by(AuditLog.created_at.desc())
                    .limit(limit)
                ).all()
            )
