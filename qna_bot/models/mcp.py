"""SQLAlchemy 2.x ORM models for user-scoped MCP servers, permissions, vault secrets, approvals, and logs."""

from datetime import datetime
from typing import Optional
import uuid

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from qna_bot.models.base import Base, utc_now


class MCPServer(Base):
    """Per-user configured MCP server matching existing mcp_servers schema with user FK."""

    __tablename__ = "mcp_servers"
    __table_args__ = (
        Index("idx_mcp_servers_user", "user_id", "name"),
        Index("idx_mcp_servers_user_enabled", "user_id", "enabled"),
    )

    server_id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: f"mcp_{uuid.uuid4().hex[:12]}")
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    transport: Mapped[str] = mapped_column(String(32), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    command: Mapped[str] = mapped_column(Text, nullable=False, default="")
    args_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    env_refs_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    url: Mapped[str] = mapped_column(Text, nullable=False, default="")
    auth_type: Mapped[str] = mapped_column(String(32), nullable=False, default="none")
    auth_header_name: Mapped[str] = mapped_column(String(64), nullable=False, default="Authorization")
    auth_env_var: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    allow_insecure_http: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    allow_private_network: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    allowed_roots_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="disconnected")
    last_connected_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    last_error: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[float] = mapped_column(Float, nullable=False, default=lambda: utc_now().timestamp())
    updated_at: Mapped[float] = mapped_column(Float, nullable=False, default=lambda: utc_now().timestamp())
    registry_source: Mapped[str] = mapped_column(String(64), nullable=False, default="manual")
    registry_id: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    listing_url: Mapped[str] = mapped_column(Text, nullable=False, default="")
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    category: Mapped[str] = mapped_column(String(64), nullable=False, default="General")
    icon_url: Mapped[str] = mapped_column(Text, nullable=False, default="")
    package_type: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    install_hint: Mapped[str] = mapped_column(Text, nullable=False, default="")
    discovered_tools_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    last_health_status: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    health_checked_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class MCPToolPermission(Base):
    """Per-user per-tool enablement and HITL permission override."""

    __tablename__ = "mcp_tool_permissions"

    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    server_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("mcp_servers.server_id", ondelete="CASCADE"), primary_key=True
    )
    tool_name: Mapped[str] = mapped_column(String(160), primary_key=True)
    category: Mapped[str] = mapped_column(String(32), nullable=False, default="MODIFICATION")
    auto_approve: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    updated_at: Mapped[float] = mapped_column(Float, nullable=False, default=lambda: utc_now().timestamp())


class MCPCredential(Base):
    """Fernet-encrypted MCP tokens/API keys/OAuth payloads isolated by (user_id, server_id, auth_type)."""

    __tablename__ = "mcp_credentials"
    __table_args__ = (
        UniqueConstraint("user_id", "server_id", "auth_type", name="uq_mcp_creds_user_srv_type"),
        Index("idx_mcp_creds_user_srv", "user_id", "server_id"),
    )

    credential_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    server_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("mcp_servers.server_id", ondelete="CASCADE"), nullable=False
    )
    auth_type: Mapped[str] = mapped_column(String(32), nullable=False)
    header_name: Mapped[str] = mapped_column(String(64), nullable=False, default="Authorization")
    encrypted_payload: Mapped[str] = mapped_column(Text, nullable=False)
    key_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    expires_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    scopes: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[float] = mapped_column(Float, nullable=False, default=lambda: utc_now().timestamp())
    updated_at: Mapped[float] = mapped_column(Float, nullable=False, default=lambda: utc_now().timestamp())


class MCPApproval(Base):
    """Backend-enforced HITL approval bound to (user_id, thread_id, server_id, tool_name, arguments_hash)."""

    __tablename__ = "mcp_approvals"
    __table_args__ = (Index("idx_mcp_approvals_user", "user_id", "thread_id", "status"),)

    approval_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    thread_id: Mapped[str] = mapped_column(String(128), nullable=False)
    server_id: Mapped[str] = mapped_column(String(64), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(160), nullable=False)
    arguments_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    masked_arguments: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PENDING")
    created_at: Mapped[float] = mapped_column(Float, nullable=False, default=lambda: utc_now().timestamp())
    decided_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class MCPExecutionLog(Base):
    """Execution telemetry log for MCP tool calls scoped to user and conversation."""

    __tablename__ = "mcp_execution_logs"
    __table_args__ = (
        Index("idx_mcp_exec_user_created", "user_id", "created_at"),
        Index("idx_mcp_exec_conv", "conversation_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: f"mexec_{uuid.uuid4().hex[:14]}")
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    conversation_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True
    )
    server_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("mcp_servers.server_id", ondelete="SET NULL"), nullable=True
    )
    tool_name: Mapped[str] = mapped_column(String(160), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    execution_time: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    error_message: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)


class MCPAuditLog(Base):
    """Audit log table for MCP tool executions with masked argument payloads."""

    __tablename__ = "mcp_audit_logs"
    __table_args__ = (Index("idx_mcp_audit_user", "user_id", "started_at"),)

    execution_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    thread_id: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    server_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    server_name: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    tool_name: Mapped[str] = mapped_column(String(160), nullable=False, default="")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="SUCCESS")
    masked_arguments: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    result_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    error: Mapped[str] = mapped_column(Text, nullable=False, default="")
    started_at: Mapped[float] = mapped_column(Float, nullable=False, default=lambda: utc_now().timestamp())
    completed_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    duration_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class MCPRegistryCache(Base):
    """Cached Official MCP Registry & Glama marketplace API responses."""

    __tablename__ = "mcp_registry_cache"

    cache_key: Mapped[str] = mapped_column(String(255), primary_key=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    fetched_at: Mapped[float] = mapped_column(Float, nullable=False)
    ttl_sec: Mapped[int] = mapped_column(Integer, nullable=False, default=300)
