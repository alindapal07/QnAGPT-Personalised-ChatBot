"""Tool permission classification and backend policy enforcement."""

from typing import Optional

from qna_bot.database.models import ToolPermissionRecord
from qna_bot.mcp.models import DiscoveredTool, PermissionCategory

_BLOCKED_PATTERNS = ("format_disk", "drop_database", "shutdown_system")

_SENSITIVE_KEYWORDS = (
    "delete",
    "remove",
    "drop",
    "destroy",
    "purge",
    "exec",
    "shell",
    "command",
    "bash",
    "powershell",
    "script",
    "chmod",
    "chown",
    "grant",
    "revoke",
    "truncate",
)

_MODIFICATION_KEYWORDS = (
    "write",
    "create",
    "update",
    "edit",
    "modify",
    "insert",
    "patch",
    "put",
    "post",
    "mkdir",
    "append",
    "move",
    "rename",
    "commit",
    "push",
    "save",
    "set_",
)

_READ_ONLY_KEYWORDS = (
    "read",
    "list",
    "get",
    "search",
    "find",
    "fetch",
    "query",
    "select",
    "inspect",
    "describe",
    "status",
    "ping",
    "view",
    "show",
    "count",
    "check",
)


def infer_permission_category(
    tool_name: str,
    read_only_hint: Optional[bool] = None,
    destructive_hint: Optional[bool] = None,
) -> PermissionCategory:
    """Classify an MCP tool safely into READ_ONLY, MODIFICATION, SENSITIVE, or BLOCKED."""
    lowered = (tool_name or "").strip().lower()

    if any(pat in lowered for pat in _BLOCKED_PATTERNS):
        return "BLOCKED"
    if destructive_hint is True or any(kw in lowered for kw in _SENSITIVE_KEYWORDS):
        return "SENSITIVE"
    if any(kw in lowered for kw in _MODIFICATION_KEYWORDS):
        return "MODIFICATION"
    if read_only_hint is True or any(kw in lowered for kw in _READ_ONLY_KEYWORDS):
        return "READ_ONLY"
    # Safe default for unknown tools: require approval
    return "MODIFICATION"


def resolve_tool_policy(
    tool: DiscoveredTool,
    override: Optional[ToolPermissionRecord] = None,
) -> tuple[PermissionCategory, bool]:
    """Return (effective_category, requires_human_approval)."""
    category: PermissionCategory = (
        override.category  # type: ignore[assignment]
        if override is not None
        else tool.permission_category
    )
    if category == "BLOCKED":
        return "BLOCKED", True
    if category == "SENSITIVE":
        return "SENSITIVE", True
    if category == "MODIFICATION":
        auto = bool(override.auto_approve) if override is not None else False
        return "MODIFICATION", not auto
    # READ_ONLY
    auto_read = bool(override.auto_approve) if override is not None else True
    return "READ_ONLY", not auto_read
