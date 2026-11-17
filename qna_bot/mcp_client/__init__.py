"""MCP Client subsystem exports."""

from typing import Any, Dict, List, Tuple

from qna_bot import resolve_lazy_export

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "AsyncLoopRunner": ("qna_bot.mcp_client.connection", "AsyncLoopRunner"),
    "AuthType": ("qna_bot.mcp_client.models", "AuthType"),
    "CredentialVault": ("qna_bot.mcp_client.credentials", "CredentialVault"),
    "DiscoveredPrompt": ("qna_bot.mcp_client.models", "DiscoveredPrompt"),
    "DiscoveredResource": ("qna_bot.mcp_client.models", "DiscoveredResource"),
    "DiscoveredTool": ("qna_bot.mcp_client.models", "DiscoveredTool"),
    "EncryptedSQLiteTokenStorage": ("qna_bot.mcp_client.credentials", "EncryptedSQLiteTokenStorage"),
    "MCPApprovalRequiredError": ("qna_bot.mcp_client.exceptions", "MCPApprovalRequiredError"),
    "MCPClientManager": ("qna_bot.mcp_client.client_manager", "MCPClientManager"),
    "MCPConnectionError": ("qna_bot.mcp_client.exceptions", "MCPConnectionError"),
    "MCPConnectionHandle": ("qna_bot.mcp_client.connection", "MCPConnectionHandle"),
    "MCPError": ("qna_bot.mcp_client.exceptions", "MCPError"),
    "MCPPromptManager": ("qna_bot.mcp_client.prompt_manager", "MCPPromptManager"),
    "MCPResourceManager": ("qna_bot.mcp_client.resource_manager", "MCPResourceManager"),
    "MCPSecurityError": ("qna_bot.mcp_client.exceptions", "MCPSecurityError"),
    "MCPServerRegistry": ("qna_bot.mcp_client.server_registry", "MCPServerRegistry"),
    "MCPServerStatus": ("qna_bot.mcp_client.models", "MCPServerStatus"),
    "MCPToolExecutionError": ("qna_bot.mcp_client.exceptions", "MCPToolExecutionError"),
    "MCPToolRegistry": ("qna_bot.mcp_client.tool_registry", "MCPToolRegistry"),
    "OAuthFlowManager": ("qna_bot.mcp_client.oauth", "OAuthFlowManager"),
    "PendingOAuthSession": ("qna_bot.mcp_client.oauth", "PendingOAuthSession"),
    "PermissionCategory": ("qna_bot.mcp_client.models", "PermissionCategory"),
    "ToolExecutionResult": ("qna_bot.mcp_client.models", "ToolExecutionResult"),
    "build_stdio_spec_from_package": ("qna_bot.mcp_client.security", "build_stdio_spec_from_package"),
    "format_masked_arguments": ("qna_bot.mcp_client.security", "format_masked_arguments"),
    "infer_permission_category": ("qna_bot.mcp_client.permissions", "infer_permission_category"),
    "make_namespaced_tool_name": ("qna_bot.mcp_client.tool_registry", "make_namespaced_tool_name"),
    "mask_secrets": ("qna_bot.mcp_client.security", "mask_secrets"),
    "resolve_tool_policy": ("qna_bot.mcp_client.permissions", "resolve_tool_policy"),
    "sanitize_tool_output": ("qna_bot.mcp_client.security", "sanitize_tool_output"),
    "validate_remote_url": ("qna_bot.mcp_client.security", "validate_remote_url"),
    "validate_stdio_config": ("qna_bot.mcp_client.security", "validate_stdio_config"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    return resolve_lazy_export(__name__, globals(), _EXPORTS, name)


def __dir__() -> List[str]:
    return sorted(__all__)
