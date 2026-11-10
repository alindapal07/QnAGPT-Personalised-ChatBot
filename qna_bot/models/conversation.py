"""SQLAlchemy 2.x ORM models for conversations, messages, feedback, and LangGraph agent executions."""

from datetime import datetime
from typing import Optional
import uuid

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from qna_bot.models.base import Base, utc_now


class Conversation(Base):
    """User-owned conversation thread."""

    __tablename__ = "conversations"
    __table_args__ = (
        Index("idx_conversations_user_order", "user_id", "status", "is_pinned", "updated_at"),
    )

    id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: f"conversation_{uuid.uuid4().hex[:12]}"
    )
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False, default="New Conversation")
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    model_name: Mapped[str] = mapped_column(String(120), nullable=False, default="openai/gpt-oss-20b")
    agent_mode: Mapped[str] = mapped_column(String(64), nullable=False, default="Auto (Agentic)")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    is_pinned: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_custom_title: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )
    archived_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="Message.id"
    )
    executions: Mapped[list["AgentExecution"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )


class Message(Base):
    """Individual message within a user-owned conversation."""

    __tablename__ = "messages"
    __table_args__ = (
        Index("idx_messages_conv_user_created", "conversation_id", "user_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    message_type: Mapped[str] = mapped_column(String(32), nullable=False, default="text")
    token_usage: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")


class AgentExecution(Base):
    """Tracked LangGraph / Agent Workspace run linked to a user and conversation."""

    __tablename__ = "agent_executions"
    __table_args__ = (
        Index("idx_agent_executions_user_started", "user_id", "started_at"),
        Index("idx_agent_executions_conv", "conversation_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: f"exec_{uuid.uuid4().hex[:14]}")
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    conversation_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True
    )
    execution_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    current_step: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    execution_metadata: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    conversation: Mapped[Optional["Conversation"]] = relationship(back_populates="executions")


class FeedbackRecord(Base):
    """User satisfaction feedback linked to an assistant message."""

    __tablename__ = "feedback_records"
    __table_args__ = (
        Index("idx_feedback_user_msg", "user_id", "message_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: f"fb_{uuid.uuid4().hex[:12]}")
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    message_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    rating: Mapped[str] = mapped_column(String(16), nullable=False)
    comment: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
