"""Pytest suite for MCP STDIO allowlists, SSRF protections, secret masking, and tool permissions."""

from pathlib import Path
import socket
from unittest.mock import patch

import pytest

from qna_bot.database.models import ToolPermissionRecord
from qna_bot.mcp.exceptions import MCPSecurityError
from qna_bot.mcp.models import DiscoveredTool
from qna_bot.mcp.permissions import infer_permission_category, resolve_tool_policy
from qna_bot.mcp.security import (
    format_masked_arguments,
    mask_secrets,
    sanitize_tool_output,
    validate_remote_url,
    validate_stdio_config,
)


def test_stdio_security_allowlist_metacharacters_roots_and_env_isolation(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_supersecretkey12345")
    monkeypatch.setenv("SAFE_MCP_VAR", "allowed_value")

    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    inside_file = docs_root / "notes.txt"

    cmd, args, env = validate_stdio_config(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", str(inside_file)],
        env_refs={"MY_VAR": "env:SAFE_MCP_VAR", "LEAK": "env:GROQ_API_KEY"},
        allowed_roots=[str(docs_root)],
    )
    assert cmd == "npx"
    assert str(inside_file) in args
    assert env["MY_VAR"] == "allowed_value"
    assert "GROQ_API_KEY" not in env
    assert "LEAK" not in env

    # Disallowed executable
    with pytest.raises(MCPSecurityError):
        validate_stdio_config("powershell", ["-Command", "whoami"])

    # Shell metacharacters rejected
    with pytest.raises(MCPSecurityError):
        validate_stdio_config("python", ["-c", "print(1); rm -rf /"])

    # Path outside allowed_roots rejected
    outside = tmp_path / "secrets" / "key.txt"
    with pytest.raises(MCPSecurityError):
        validate_stdio_config(
            "npx",
            [str(outside)],
            allowed_roots=[str(docs_root)],
        )


def test_remote_url_ssrf_and_https_protections() -> None:
    # HTTP blocked by default
    with pytest.raises(MCPSecurityError):
        validate_remote_url("http://mcp.example.com/mcp")

    # Localhost and private/link-local IPs blocked
    for bad_url in (
        "https://localhost/mcp",
        "https://127.0.0.1/mcp",
        "https://10.0.0.5/mcp",
        "https://192.168.1.10/mcp",
        "https://169.254.169.254/latest/meta-data",
        "https://[::1]/mcp",
    ):
        with pytest.raises(MCPSecurityError):
            validate_remote_url(bad_url)

    # DNS resolving to private IP blocked
    with patch(
        "qna_bot.mcp.security.socket.getaddrinfo",
        return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.12.0.5", 443))],
    ):
        with pytest.raises(MCPSecurityError):
            validate_remote_url("https://internal.corp.example/mcp")

    # Valid public HTTPS passes when DNS resolves to public IP
    with patch(
        "qna_bot.mcp.security.socket.getaddrinfo",
        return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))],
    ):
        assert validate_remote_url("https://mcp.example.com/mcp") == "https://mcp.example.com/mcp"


def test_secret_masking_and_output_sanitization() -> None:
    masked = mask_secrets(
        {
            "api_key": "sk-1234567890abcdef",
            "path": "/docs/readme.md",
            "nested": {"authorization": "Bearer abcdefghijklmnop"},
        }
    )
    assert masked["api_key"] == "***REDACTED***"
    assert masked["nested"]["authorization"] == "***REDACTED***"
    assert masked["path"] == "/docs/readme.md"

    formatted = format_masked_arguments({"token": "secret-val", "query": "langgraph"})
    assert "secret-val" not in formatted
    assert "***REDACTED***" in formatted

    sanitized = sanitize_tool_output("Header Bearer gsk_1234567890abcdef1234\x00\n" + ("x" * 200), max_chars=80)
    assert "gsk_1234567890" not in sanitized
    assert "\x00" not in sanitized
    assert "truncated" in sanitized


def test_permission_inference_and_policy_resolution() -> None:
    assert infer_permission_category("read_file") == "READ_ONLY"
    assert infer_permission_category("write_file") == "MODIFICATION"
    assert infer_permission_category("delete_repository") == "SENSITIVE"
    assert infer_permission_category("format_disk") == "BLOCKED"
    assert infer_permission_category("mystery_action") == "MODIFICATION"

    tool = DiscoveredTool(
        namespaced_name="fs__write_file",
        original_name="write_file",
        server_id="s1",
        server_name="fs",
        user_id="alice",
        description="Write file",
        permission_category="MODIFICATION",
        auto_approve=False,
    )
    cat, requires_approval = resolve_tool_policy(tool)
    assert cat == "MODIFICATION"
    assert requires_approval is True

    # SENSITIVE cannot be auto-approved even if override attempts it
    sens_tool = DiscoveredTool(
        namespaced_name="fs__delete_file",
        original_name="delete_file",
        server_id="s1",
        server_name="fs",
        user_id="alice",
        description="Delete file",
        permission_category="SENSITIVE",
        auto_approve=False,
    )
    override = ToolPermissionRecord("alice", "s1", "delete_file", "SENSITIVE", auto_approve=True)
    cat2, req2 = resolve_tool_policy(sens_tool, override)
    assert cat2 == "SENSITIVE"
    assert req2 is True
