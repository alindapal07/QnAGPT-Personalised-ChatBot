"""MCP client and security exception hierarchy."""

from qna_bot.core.exceptions import QnABotError


class MCPError(QnABotError):
    """Base exception for Model Context Protocol client errors."""


class MCPConnectionError(MCPError):
    """Raised when connecting, initializing, or communicating with an MCP server fails."""


class MCPSecurityError(MCPError):
    """Raised when an MCP server configuration, URL, or command violates security policy."""


class MCPToolExecutionError(MCPError):
    """Raised when an MCP tool call fails, times out, or returns invalid output."""


class MCPApprovalRequiredError(MCPError):
    """Raised when a tool execution requires explicit human approval before proceeding."""
