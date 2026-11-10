"""SQLite persistence models for threads, MCP servers, permissions, approvals, and audit logs."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MCPServerRecord:
    """Persisted MCP server configuration scoped to a user."""

    server_id: str
    user_id: str
    name: str
    transport: str  # "stdio" | "streamable_http" | "sse"
    enabled: bool = True
    command: str = ""
    args: List[str] = field(default_factory=list)
    env_refs: Dict[str, str] = field(default_factory=dict)
    url: str = ""
    auth_type: str = "none"  # "none" | "bearer_env" | "api_key_env" | "bearer_token" | "api_key" | "oauth"
    auth_header_name: str = "Authorization"
    auth_env_var: str = ""
    allow_insecure_http: bool = False
    allow_private_network: bool = False
    allowed_roots: List[str] = field(default_factory=list)
    status: str = "disconnected"
    last_connected_at: Optional[float] = None
    last_error: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    registry_source: str = "manual"
    registry_id: str = ""
    listing_url: str = ""
    description: str = ""
    category: str = "General"
    icon_url: str = ""
    package_type: str = ""
    install_hint: str = ""
    discovered_tools: List[Dict[str, Any]] = field(default_factory=list)
    last_health_status: str = "unknown"
    health_checked_at: Optional[float] = None


@dataclass
class ToolPermissionRecord:
    """Per-user tool permission and activation override."""

    user_id: str
    server_id: str
    tool_name: str
    category: str  # "READ_ONLY" | "MODIFICATION" | "SENSITIVE" | "BLOCKED"
    auto_approve: bool = False
    updated_at: float = 0.0
    enabled: bool = True


@dataclass
class EncryptedCredentialRecord:
    """Encrypted MCP credential record isolated by (user_id, server_id, auth_type)."""

    credential_id: str
    user_id: str
    server_id: str
    auth_type: str  # "bearer_token" | "api_key" | "oauth_token" | "oauth_client" | "env_secrets"
    header_name: str = "Authorization"
    encrypted_payload: str = ""
    key_fingerprint: str = ""
    expires_at: Optional[float] = None
    scopes: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0

    def __repr__(self) -> str:
        return (
            f"EncryptedCredentialRecord(credential_id={self.credential_id!r}, "
            f"user_id={self.user_id!r}, server_id={self.server_id!r}, "
            f"auth_type={self.auth_type!r}, encrypted_payload='***REDACTED***')"
        )


@dataclass
class ApprovalRecord:
    """Persisted human-in-the-loop approval record bound to exact arguments."""

    approval_id: str
    user_id: str
    thread_id: str
    server_id: str
    tool_name: str
    arguments_hash: str
    masked_arguments: str
    category: str
    reason: str
    status: str = "PENDING"  # "PENDING" | "APPROVED" | "REJECTED" | "CONSUMED"
    created_at: float = 0.0
    decided_at: Optional[float] = None


@dataclass
class AuditLogRecord:
    """Execution audit entry for MCP tool invocations."""

    execution_id: str
    user_id: str
    thread_id: str
    server_id: str
    server_name: str
    tool_name: str
    status: str  # "completed" | "failed" | "rejected" | "awaiting_approval" | "blocked"
    masked_arguments: str
    result_summary: str
    error: str = ""
    started_at: float = 0.0
    completed_at: Optional[float] = None
    duration_ms: Optional[float] = None
