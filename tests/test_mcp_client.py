"""Pytest suite for MCPClientManager, dynamic server registry, resource/prompt discovery, and execution."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from qna_bot.database.connection import DatabaseManager
from qna_bot.mcp.client_manager import MCPClientManager
from qna_bot.mcp.exceptions import MCPConnectionError, MCPSecurityError, MCPToolExecutionError


class _FakeMCPSession:
    def __init__(self, fail_tool: bool = False) -> None:
        self.fail_tool = fail_tool

    async def list_tools(self):
        return SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name="list_tables",
                    description="List database tables",
                    input_schema={"type": "object", "properties": {}},
                    annotations=SimpleNamespace(readOnlyHint=True, destructiveHint=False),
                ),
                SimpleNamespace(
                    name="read_query",
                    description="Execute read-only SQL query",
                    input_schema={
                        "type": "object",
                        "properties": {"sql": {"type": "string"}},
                        "required": ["sql"],
                    },
                    annotations=SimpleNamespace(readOnlyHint=True, destructiveHint=False),
                ),
            ]
        )

    async def list_resources(self):
        return SimpleNamespace(
            resources=[
                SimpleNamespace(
                    uri="db://schema",
                    name="Database Schema",
                    description="Public schema",
                    mime_type="application/json",
                )
            ]
        )

    async def list_prompts(self):
        return SimpleNamespace(
            prompts=[
                SimpleNamespace(
                    name="sql_helper",
                    description="SQL query prompt",
                    arguments=[SimpleNamespace(name="table", description="Table name", required=True)],
                )
            ]
        )

    async def call_tool(self, name: str, arguments=None):
        if self.fail_tool:
            return SimpleNamespace(
                content=[SimpleNamespace(text="Query failed: syntax error")],
                is_error=True,
                structured_content=None,
            )
        return SimpleNamespace(
            content=[SimpleNamespace(text=f"Executed {name} with {arguments}")],
            is_error=False,
            structured_content={"ok": True},
        )

    async def read_resource(self, uri: str):
        return SimpleNamespace(
            contents=[SimpleNamespace(text='{"tables": ["users"]}', mime_type="application/json")]
        )

    async def get_prompt(self, name: str, arguments=None):
        return SimpleNamespace(
            description="SQL helper prompt",
            messages=[SimpleNamespace(role="user", content=SimpleNamespace(text="Inspect table users"))],
        )

    async def send_ping(self):
        return {}


class _MockHandle:
    """Zero-socket connection handle executing async session coroutines synchronously."""

    def __init__(self, record, loop_runner, connect_timeout=15, session_factory=None) -> None:
        self.record = record
        self._session_factory = session_factory
        self._session = None

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    @staticmethod
    def _step(coro):
        try:
            coro.send(None)
        except StopIteration as stop:
            return stop.value
        raise RuntimeError("Mock coroutine did not resolve synchronously.")

    def connect(self) -> None:
        try:
            self._session = self._step(self._session_factory(self.record, None))
        except Exception as exc:
            raise MCPConnectionError(f"Failed to connect to '{self.record.name}': {exc}") from exc

    def disconnect(self) -> None:
        self._session = None

    def run_call(self, coro_fn, timeout: float = 30.0):
        return self._step(coro_fn(self._session))


def test_mcp_client_lifecycle_discovery_execution_and_user_isolation(tmp_path: Path) -> None:
    db = DatabaseManager(db_path=tmp_path / "mcp.db")
    fake_session = _FakeMCPSession()

    async def _fake_factory(record, stack):
        return fake_session

    mgr = MCPClientManager(db_manager=db, session_factory=_fake_factory)
    try:
        with patch("qna_bot.mcp.client_manager.MCPConnectionHandle", _MockHandle):
            rec = mgr.register_server(
                user_id="alice",
                name="SQLite Analytics",
                transport="stdio",
                command="uvx",
                args=["mcp-server-sqlite", "--db-path", "demo.db"],
            )
            assert rec.name == "SQLite Analytics"

            # Bob cannot access or connect Alice's server
            assert mgr.get_all_server_statuses("bob") == []
            with pytest.raises(Exception):
                mgr.connect_server(rec.server_id, user_id="bob")

            # Connect & discover capabilities
            status = mgr.test_connection(rec.server_id, user_id="alice")
            assert status.connected is True
            assert status.tool_count == 2
            assert status.resource_count == 1
            assert status.prompt_count == 1

            tools = mgr.list_available_tools("alice")
            assert [t.namespaced_name for t in tools] == [
                "sqlite_analytics__list_tables",
                "sqlite_analytics__read_query",
            ]

            # Execute tool
            exec_res = mgr.execute_tool(
                user_id="alice",
                namespaced_tool_name="sqlite_analytics__read_query",
                arguments={"sql": "SELECT 1"},
            )
            assert exec_res.success is True
            assert "read_query" in exec_res.content

            # Read resource & prompt
            res_out = mgr.read_resource("alice", rec.server_id, "db://schema")
            assert res_out["mime_type"] == "application/json"
            assert "users" in res_out["content"]

            prm_out = mgr.get_prompt("alice", rec.server_id, "sql_helper", {"table": "users"})
            assert prm_out["messages"][0]["content"] == "Inspect table users"

            # Disconnect clears discovered capabilities cleanly
            mgr.disconnect_server(rec.server_id, "alice")
            assert mgr.list_available_tools("alice") == []
            assert mgr.get_server_status(rec.server_id, "alice").connected is False

            # Reconnect restores capabilities
            status2 = mgr.reconnect_server(rec.server_id, "alice")
            assert status2.connected is True
            assert status2.tool_count == 2

            # Remove server
            assert mgr.remove_server(rec.server_id, "alice") is True
            assert mgr.get_all_server_statuses("alice") == []
    finally:
        db.close()


def test_mcp_client_handles_connection_and_tool_errors(tmp_path: Path) -> None:
    db = DatabaseManager(db_path=tmp_path / "mcp_err.db")

    async def _failing_factory(record, stack):
        raise ConnectionRefusedError("Server down")

    mgr = MCPClientManager(db_manager=db, session_factory=_failing_factory)
    try:
        with patch("qna_bot.mcp.client_manager.MCPConnectionHandle", _MockHandle):
            with pytest.raises(MCPSecurityError):
                mgr.register_server("alice", "Bad Local", "stdio", command="cmd", args=["/c", "dir"])

            rec = mgr.register_server("alice", "Down Server", "stdio", command="python", args=["-m", "fake"])
            with pytest.raises(MCPConnectionError):
                mgr.connect_server(rec.server_id, "alice")

            status = mgr.get_server_status(rec.server_id, "alice")
            assert status.connected is False
            assert status.status == "error"
            assert "Server down" in status.last_error

            with pytest.raises(MCPToolExecutionError):
                mgr.execute_tool("alice", "down_server__missing", {})
    finally:
        db.close()
