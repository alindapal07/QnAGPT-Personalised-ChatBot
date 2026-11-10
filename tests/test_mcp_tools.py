"""Pytest suite for MCP tool discovery, JSON Schema validation, LangChain adapter, and user isolation."""

from pathlib import Path
from types import SimpleNamespace

from pydantic import ValidationError
import pytest

from qna_bot.agent.mcp_adapter import (
    build_mcp_langchain_tools,
    build_pydantic_args_schema,
    validate_tool_arguments,
)
from qna_bot.database.connection import DatabaseManager
from qna_bot.mcp.client_manager import MCPClientManager
from qna_bot.mcp.models import DiscoveredTool
from qna_bot.mcp.tool_registry import make_namespaced_tool_name


def test_namespaced_tool_naming_and_schema_validation() -> None:
    assert make_namespaced_tool_name("GitHub MCP", "search_repositories") == "github_mcp__search_repositories"

    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    }
    model_cls = build_pydantic_args_schema("github__search", schema)
    parsed = model_cls(query="langgraph")
    assert parsed.query == "langgraph"
    assert parsed.limit == 5

    tool = DiscoveredTool(
        namespaced_name="github__search",
        original_name="search",
        server_id="srv1",
        server_name="GitHub",
        user_id="alice",
        description="Search repos",
        input_schema=schema,
        permission_category="READ_ONLY",
        auto_approve=True,
    )
    assert validate_tool_arguments(tool, {"query": "rag", "limit": 3}) == {"query": "rag", "limit": 3}

    with pytest.raises(ValidationError):
        validate_tool_arguments(tool, {"limit": "not_an_int"})


def test_mcp_langchain_adapter_filters_blocked_and_isolates_users(tmp_path: Path) -> None:
    db = DatabaseManager(db_path=tmp_path / "mcp_tools.db")
    mgr = MCPClientManager(db_manager=db)
    try:
        mgr.tool_registry.register_server_tools(
            user_id="alice",
            server_id="srv_fs",
            server_name="Filesystem",
            raw_tools=[
                SimpleNamespace(
                    name="read_file",
                    description="Read file contents",
                    input_schema={
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                    annotations=SimpleNamespace(readOnlyHint=True, destructiveHint=False),
                ),
                SimpleNamespace(
                    name="format_disk",
                    description="Dangerous action",
                    input_schema={"type": "object", "properties": {}},
                    annotations=SimpleNamespace(readOnlyHint=False, destructiveHint=True),
                ),
            ],
        )

        alice_tools = build_mcp_langchain_tools(mgr, user_id="alice")
        assert [t.name for t in alice_tools] == ["filesystem__read_file"]
        assert alice_tools[0].metadata["mcp_server_id"] == "srv_fs"
        assert alice_tools[0].metadata["permission_category"] == "READ_ONLY"

        # Bob has zero access to Alice's tools
        assert build_mcp_langchain_tools(mgr, user_id="bob") == []
    finally:
        db.close()
