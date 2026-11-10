"""Tests for encrypted credential vault, OAuth PKCE flow, HTTP auth, health, and tool toggles."""

from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
import socket
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from langchain_core.tools import StructuredTool
import pytest

from qna_bot.agent.graph import build_agent_graph, clear_compiled_graph_cache
from qna_bot.agent.mcp_adapter import build_mcp_langchain_tools
from qna_bot.agent.nodes import _select_relevant_tools_for_binding
from qna_bot.database.connection import DatabaseManager
from qna_bot.database.models import MCPServerRecord
from qna_bot.mcp.client_manager import MCPClientManager
from qna_bot.mcp.connection import AsyncLoopRunner, MCPConnectionHandle, _format_exception_detail, resolve_auth_headers
from qna_bot.mcp.credentials import CredentialVault
from qna_bot.mcp.exceptions import MCPSecurityError
from qna_bot.mcp.oauth import OAuthFlowManager
from qna_bot.services.mcp_service import MCPService
from tests.test_mcp_client import _FakeMCPSession, _MockHandle


def test_credential_vault_encrypts_at_rest_and_enforces_user_isolation(tmp_path: Path) -> None:
    db = DatabaseManager(db_path=tmp_path / "vault.db")
    try:
        vault = CredentialVault(db_manager=db, encryption_key="test-passphrase-for-fernet")
        vault.store_bearer_token("alice", "srv_1", "tok_live_secret_xyz123")
        vault.store_api_key("bob", "srv_1", "api_bob_secret_999")

        # Raw SQLite storage contains ciphertext only (never plaintext secret substrings)
        rows = db.connection.execute("SELECT user_id, encrypted_payload FROM mcp_credentials").fetchall()
        assert len(rows) == 2
        for row in rows:
            assert "tok_live_secret_xyz123" not in row["encrypted_payload"]
            assert "api_bob_secret_999" not in row["encrypted_payload"]
            assert "gAAAA" in row["encrypted_payload"]

        # Safe status view never leaks secrets
        status = vault.get_credential_status("alice", "srv_1")
        assert status["configured"] is True
        assert "tok_live_secret_xyz123" not in repr(status)

        # User isolation prevents cross-user reads
        assert vault.load_secret("bob", "srv_1", "bearer_token") is None
        assert vault.load_secret("alice", "srv_1", "bearer_token") == {"token": "tok_live_secret_xyz123"}

        # resolve_auth_headers reads encrypted tokens cleanly
        mgr = MCPClientManager(db_manager=db)
        with patch(
            "qna_bot.mcp.security.socket.getaddrinfo",
            return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))],
        ):
            rec = mgr.register_server(
                user_id="alice",
                server_id="srv_1",
                name="Remote API",
                transport="streamable_http",
                url="https://api.example.com/mcp",
                auth_type="bearer_token",
            )
            assert resolve_auth_headers(rec, vault=vault) == {
                "Authorization": "Bearer tok_live_secret_xyz123"
            }
        mgr.shutdown()
    finally:
        db.close()


def test_oauth_pkce_state_validation_exchange_refresh_revocation_and_ssrf(tmp_path: Path) -> None:
    db = DatabaseManager(db_path=tmp_path / "oauth.db")
    try:
        vault = CredentialVault(db_manager=db, encryption_key="oauth-secret-key")
        oauth = OAuthFlowManager(vault=vault, default_redirect_uri="http://localhost:8501/oauth/callback")

        with patch(
            "qna_bot.mcp.security.socket.getaddrinfo",
            return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))],
        ):
            # Block SSRF metadata discovery returning private/loopback auth endpoints
            with pytest.raises(MCPSecurityError):
                oauth.discover_metadata(
                    "https://mcp.example.com/mcp",
                    http_get=lambda url: {
                        "authorization_endpoint": "http://169.254.169.254/latest/meta-data",
                        "token_endpoint": "https://auth.example.com/token",
                    },
                )

            session = oauth.start_authorization_flow(
                user_id="alice",
                server_id="srv_oauth",
                server_url="https://mcp.example.com/mcp",
                client_id="client_abc",
                authorization_endpoint="https://auth.example.com/authorize",
                token_endpoint="https://auth.example.com/token",
                revocation_endpoint="https://auth.example.com/revoke",
                scopes="read write",
            )
            qs = parse_qs(urlparse(session.authorization_url).query)
            assert qs["code_challenge_method"] == ["S256"]
            assert qs["state"] == [session.state]
            assert qs["client_id"] == ["client_abc"]

            # Reject forged state (CSRF check)
            with pytest.raises(MCPSecurityError, match="state validation failed"):
                oauth.complete_authorization_callback(
                    user_id="alice",
                    server_id="srv_oauth",
                    callback_url_or_code="https://localhost:8501/oauth/callback?code=auth123&state=forged",
                )

            captured_forms: list[tuple[str, dict]] = []

            def _fake_post(url: str, form: dict) -> dict:
                captured_forms.append((url, form))
                if form.get("grant_type") == "authorization_code":
                    return {
                        "access_token": "oauth_tok_1",
                        "refresh_token": "ref_tok_1",
                        "expires_in": 3600,
                        "token_type": "Bearer",
                        "scope": "read write",
                    }
                if form.get("grant_type") == "refresh_token":
                    return {
                        "access_token": "oauth_tok_2",
                        "expires_in": 3600,
                        "token_type": "Bearer",
                    }
                return {}

            # Complete valid callback and verify PKCE code_verifier was sent
            tok1 = oauth.complete_authorization_callback(
                user_id="alice",
                server_id="srv_oauth",
                callback_url_or_code=f"http://localhost:8501/oauth/callback?code=auth123&state={session.state}",
                http_post=_fake_post,
            )
            assert tok1.access_token == "oauth_tok_1"
            assert captured_forms[0][1]["code_verifier"] == session.code_verifier

            # Refresh updates stored encrypted token
            tok2 = oauth.refresh_access_token(
                user_id="alice",
                server_id="srv_oauth",
                http_post=_fake_post,
            )
            assert tok2.access_token == "oauth_tok_2"
            assert tok2.refresh_token == "ref_tok_1"

            # Revocation calls revocation endpoint and deletes encrypted credential rows
            revoked = oauth.revoke_access("alice", "srv_oauth", http_post=_fake_post)
            assert revoked is True
            assert vault.get_credential_status("alice", "srv_oauth")["configured"] is False
    finally:
        db.close()


def test_tool_toggle_persisted_metadata_health_and_langgraph_filtering(tmp_path: Path) -> None:
    clear_compiled_graph_cache()
    db = DatabaseManager(db_path=tmp_path / "toggles.db")
    fake_session = _FakeMCPSession()

    async def _fake_factory(record, stack):
        return fake_session

    mgr = MCPClientManager(db_manager=db, session_factory=_fake_factory)
    svc = MCPService(client_manager=mgr, db_manager=db)
    try:
        with (
            patch(
                "qna_bot.mcp.security.socket.getaddrinfo",
                return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))],
            ),
            patch("qna_bot.mcp.client_manager.MCPConnectionHandle", _MockHandle),
        ):
            rec = svc.register_server(
                user_id="alice",
                name="Analytics HTTP",
                transport="streamable_http",
                url="https://analytics.example.com/mcp",
                auth_type="none",
            )
            status = svc.connect(rec.server_id, "alice")
            assert status.connected is True
            assert status.tool_count == 2
            assert status.health_status == "healthy"

            # Ping health check updates health telemetry cleanly
            checked = svc.check_health(rec.server_id, "alice")
            assert checked.health_status == "healthy"

            # Disable one tool -> excluded from LangGraph adapter, retained in include_disabled=True
            svc.set_tool_enabled("alice", rec.server_id, "read_query", False)
            active_tools = mgr.list_available_tools("alice")
            assert [t.original_name for t in active_tools] == ["list_tables"]
            all_tools = svc.list_tools("alice", include_disabled=True)
            assert [t.original_name for t in all_tools] == ["list_tables", "read_query"]

            lc_tools = build_mcp_langchain_tools(mgr, user_id="alice")
            assert [t.name for t in lc_tools] == ["analytics_http__list_tables"]

            # Re-enable tool without recreating session -> LangGraph cache key shifts automatically
            g1 = build_agent_graph(user_id="alice", api_key="test", mcp_manager=mgr, use_cache=True)
            svc.set_tool_enabled("alice", rec.server_id, "read_query", True)
            g2 = build_agent_graph(user_id="alice", api_key="test", mcp_manager=mgr, use_cache=True)
            assert g1 is not g2

            # Disconnecting preserves offline inspectable discovered metadata in SQLite
            svc.disconnect(rec.server_id, "alice")
            persisted = svc.list_tools_with_persisted("alice", rec.server_id)
            assert [t.original_name for t in persisted] == ["list_tables", "read_query"]
    finally:
        clear_compiled_graph_cache()
        mgr.shutdown()
        db.close()


def test_streamable_http_two_tuple_unpacking_exception_group_and_tool_binding_cap() -> None:
    # 1. Nested ExceptionGroup unwrapping extracts actionable leaf errors cleanly
    eg = ExceptionGroup(
        "unhandled errors in a TaskGroup",
        [ExceptionGroup("inner", [ValueError("HTTP 401 Unauthorized")])],
    )
    assert _format_exception_detail(eg) == "HTTP 401 Unauthorized"

    # 2. Verify _open_transport_session handles mcp==2.3.0 2-tuple yield from streamable_http_client
    record = MCPServerRecord(
        server_id="srv_http",
        user_id="alice",
        name="NeuralVerge",
        transport="streamable_http",
        url="https://mcp.example.com/mcp",
        auth_type="none",
    )
    handle = MCPConnectionHandle(record=record, loop_runner=AsyncLoopRunner())
    initialized = {"ok": False}

    @asynccontextmanager
    async def _fake_streamable_http(url, *, http_client=None, **kwargs):
        yield "read_stream", "write_stream"

    @asynccontextmanager
    async def _fake_http_client(**kwargs):
        yield "http_client"

    class _FakeSession:
        def __init__(self, r, w, read_timeout_seconds=15):
            assert (r, w) == ("read_stream", "write_stream")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def initialize(self):
            initialized["ok"] = True

    async def _run() -> None:
        async with AsyncExitStack() as stack:
            sess = await handle._open_transport_session(stack)
            assert sess is not None

    with (
        patch(
            "qna_bot.mcp.security.socket.getaddrinfo",
            return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))],
        ),
        patch("qna_bot.mcp.connection.create_mcp_http_client", _fake_http_client),
        patch("qna_bot.mcp.connection.streamable_http_client", _fake_streamable_http),
        patch("qna_bot.mcp.connection.ClientSession", _FakeSession),
    ):
        coro = _run()
        try:
            coro.send(None)
        except StopIteration:
            pass
    assert initialized["ok"] is True

    # 3. Tool binding cap ranks query-relevant tools first for servers with >35 tools
    many_tools = [
        StructuredTool.from_function(
            func=lambda q="": q,
            name=f"nv__tool_{i}",
            description="Generic helper" if i != 200 else "Analyze financial invoice compliance",
            metadata={"mcp_original_tool": f"tool_{i}", "permission_category": "READ_ONLY"},
        )
        for i in range(363)
    ]
    builtin = [
        StructuredTool.from_function(func=lambda q="": q, name="web_search", description="Search web")
    ]
    selected = _select_relevant_tools_for_binding(
        many_tools,
        builtin,
        user_query="check invoice compliance",
        max_mcp_tools=35,
    )
    assert len(selected) == 36
    assert any(t.name == "nv__tool_200" for t in selected)
    assert any(t.name == "web_search" for t in selected)
