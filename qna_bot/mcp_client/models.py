"""Typed MCP domain models for servers, tools, resources, prompts, and executions."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

TransportType = Literal["stdio", "streamable_http", "sse"]
AuthType = Literal["none", "bearer_env", "api_key_env", "bearer_token", "api_key", "oauth"]
PermissionCategory = Literal["READ_ONLY", "MODIFICATION", "SENSITIVE", "BLOCKED"]


@dataclass(frozen=True)
class DiscoveredTool:
    """Normalized representation of an MCP server tool."""

    namespaced_name: str
    original_name: str
    server_id: str
    server_name: str
    user_id: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    read_only_hint: Optional[bool] = None
    destructive_hint: Optional[bool] = None
    permission_category: PermissionCategory = "MODIFICATION"
    auto_approve: bool = False
    enabled: bool = True


@dataclass(frozen=True)
class DiscoveredResource:
    """Normalized representation of an MCP server resource."""

    uri: str
    name: str
    server_id: str
    server_name: str
    user_id: str
    description: str = ""
    mime_type: str = "text/plain"


@dataclass(frozen=True)
class DiscoveredPrompt:
    """Normalized representation of an MCP server prompt."""

    name: str
    server_id: str
    server_name: str
    user_id: str
    description: str = ""
    arguments: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class MCPServerStatus:
    """Live connection, auth, health, and capability status for a configured MCP server."""

    server_id: str
    name: str
    transport: str
    enabled: bool
    connected: bool
    status: str
    tool_count: int = 0
    resource_count: int = 0
    prompt_count: int = 0
    last_connected_at: Optional[float] = None
    last_error: str = ""
    registry_source: str = "manual"
    listing_url: str = ""
    auth_type: str = "none"
    credential_configured: bool = False
    credential_expired: bool = False
    health_status: str = "unknown"
    health_checked_at: Optional[float] = None


@dataclass(frozen=True)
class ToolExecutionResult:
    """Sanitized result returned from executing an MCP tool."""

    server_id: str
    server_name: str
    tool_name: str
    success: bool
    content: str
    structured_content: Optional[Any] = None
    is_error: bool = False
    duration_ms: float = 0.0
    error: str = ""
