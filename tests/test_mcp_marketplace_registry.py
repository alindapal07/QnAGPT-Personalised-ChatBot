"""Tests for dynamic MCP Registry discovery (Official Registry & Glama), caching, rate limits, and STDIO validation."""

import json
from pathlib import Path
import socket
from typing import Dict, Tuple
from unittest.mock import patch

import pytest

from qna_bot.database.connection import DatabaseManager
from qna_bot.mcp.client_manager import MCPClientManager
from qna_bot.mcp.exceptions import MCPSecurityError
from qna_bot.mcp.registry.base import CachedRegistryHTTPClient, RegistryRateLimitError
from qna_bot.mcp.registry.glama_provider import GlamaRegistryProvider
from qna_bot.mcp.registry.official_provider import OfficialMCPRegistryProvider
from qna_bot.mcp.registry.service import MCPRegistryService
from qna_bot.services.mcp_service import MCPService


def _fake_official_payload() -> dict:
    return {
        "servers": [
            {
                "server": {
                    "name": "io.github.demo/weather-http",
                    "title": "Weather Insights",
                    "description": "Live forecasts via Streamable HTTP MCP",
                    "version": "1.2.0",
                    "repository": {"url": "https://github.com/demo/weather-mcp"},
                    "remotes": [
                        {
                            "type": "streamable-http",
                            "url": "https://weather.example.com/mcp",
                            "headers": [{"name": "Authorization"}],
                        }
                    ],
                },
                "_meta": {
                    "io.modelcontextprotocol.registry/official": {
                        "publishedAt": "2026-10-01T12:00:00Z",
                        "updatedAt": "2026-11-01T08:00:00Z",
                    }
                },
            },
            {
                "server": {
                    "name": "io.github.demo/sqlite-local",
                    "title": "SQLite Explorer",
                    "description": "Local database inspection over PyPI uvx package",
                    "version": "0.4.0",
                    "packages": [
                        {
                            "registryType": "pypi",
                            "identifier": "mcp-server-sqlite",
                            "version": "0.4.0",
                            "environmentVariables": [
                                {"name": "SQLITE_READ_ONLY", "required": False, "isSecret": False}
                            ],
                        }
                    ],
                },
                "_meta": {"created_at": "2026-09-10T00:00:00Z"},
            },
            {
                "server": {
                    "name": "io.github.demo/docs-only",
                    "title": "Manual Source Repo",
                    "description": "Source-only project without remote or package",
                    "repository": {"url": "https://github.com/demo/manual-only"},
                },
            },
        ],
        "metadata": {"count": 3, "nextCursor": "cursor_page_2"},
    }


def _fake_glama_payload() -> dict:
    return {
        "connectors": [
            {
                "displayName": "Sentry Monitor",
                "namespace": "sentry",
                "slug": "sentry",
                "url": "https://glama.ai/mcp/connectors/dev.sentry.mcp/sentry",
                "description": "Track issues, stack traces, and performance metrics.",
                "connection": {
                    "authType": "oauth2",
                    "transport": "streamable_http",
                    "url": "https://mcp.sentry.example.com/mcp",
                },
                "healthy": True,
                "toolCount": 12,
                "thumbnailUrl": "https://glama.ai/mcp/connectors/dev.sentry.mcp/sentry/thumbnail.webp",
                "attributes": ["recommended", "auth:oauth2", "category:developer-tools", "status:healthy"],
                "spdxLicense": {"name": "Apache-2.0"},
                "repository": {"url": "https://github.com/getsentry/sentry-mcp"},
                "tools": [
                    {"name": "search_issues", "description": "Search Sentry issues"},
                    {"name": "get_trace", "description": "Inspect distributed trace"},
                ],
                "environmentVariablesJsonSchema": {
                    "type": "object",
                    "properties": {"SENTRY_TOKEN": {"type": "string", "description": "Auth token"}},
                    "required": ["SENTRY_TOKEN"],
                },
            }
        ],
        "pageInfo": {"hasNextPage": True, "endCursor": "glama_cursor_2"},
    }


def test_official_and_glama_registry_discovery_search_filter_and_caching(tmp_path: Path) -> None:
    calls: list[str] = []

    def _transport(url: str, headers: Dict[str, str]) -> Tuple[int, Dict[str, str], str]:
        calls.append(url)
        if "registry.modelcontextprotocol.io" in url:
            return 200, {}, json.dumps(_fake_official_payload())
        if "glama.ai" in url:
            assert headers.get("Authorization") == "Bearer glama_test_key"
            return 200, {}, json.dumps(_fake_glama_payload())
        return 404, {}, "{}"

    db = DatabaseManager(db_path=tmp_path / "reg.db")
    try:
        http = CachedRegistryHTTPClient(transport=_transport, cache_repo=None)
        reg_svc = MCPRegistryService(db_manager=db, http_client=http)
        # Re-bind cache_repo from reg_svc onto http cleanly and wire mocked Glama API key
        http.cache_repo = reg_svc.cache_repo
        reg_svc.official = OfficialMCPRegistryProvider(http_client=http)
        reg_svc.glama = GlamaRegistryProvider(http_client=http, api_key="glama_test_key")

        with patch(
            "qna_bot.mcp.security.socket.getaddrinfo",
            return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))],
        ):
            page1 = reg_svc.discover(source="all")
            assert len(page1.items) == 4
            assert page1.from_cache is False
            assert ("Glama MCP Directory", "https://glama.ai") in page1.attributions

            # Second call hits SQLite TTL cache (no new HTTP requests)
            calls_before = len(calls)
            page_cached = reg_svc.discover(source="all")
            assert len(calls) == calls_before
            assert page_cached.from_cache is True

            # Non-connectable listing is preserved accurately with connectability_reason
            docs_item = next(i for i in page1.items if "Manual Source Repo" in i.title)
            assert docs_item.direct_connectable is False
            assert docs_item.connectability_reason

            # Tool capability search finds Glama Sentry connector dynamically
            sentry_search = reg_svc.discover(source="glama", query="search_issues")
            assert len(sentry_search.items) == 1
            sentry_item = sentry_search.items[0]
            assert sentry_item.listing_url == "https://glama.ai/mcp/connectors/dev.sentry.mcp/sentry"
            assert sentry_item.direct_connectable is True
            assert sentry_item.tool_count == 12
            assert sentry_item.auth_methods == ("oauth",)
            assert sentry_item.icon_url.endswith("thumbnail.webp")
            assert sentry_search.has_next_page is True
            assert sentry_search.next_cursor == "glama_cursor_2"

            # Category and transport filtering
            db_items = reg_svc.discover(source="official_mcp", category="Data & Databases", transport_filter="stdio")
            assert [i.title for i in db_items.items] == ["SQLite Explorer"]
            assert db_items.items[0].transport_options[0].command == "uvx"

            # Official pagination cursor forwarded
            assert reg_svc.discover(source="official_mcp").next_cursor == "cursor_page_2"
    finally:
        db.close()


def test_registry_handles_rate_limit_and_api_errors_gracefully(tmp_path: Path) -> None:
    attempts = {"count": 0}

    def _rate_limited(url: str, headers: Dict[str, str]) -> Tuple[int, Dict[str, str], str]:
        attempts["count"] += 1
        return 429, {"Retry-After": "0.1"}, "{}"

    db = DatabaseManager(db_path=tmp_path / "reg_err.db")
    try:
        http = CachedRegistryHTTPClient(transport=_rate_limited)
        official = OfficialMCPRegistryProvider(http_client=http)
        with patch(
            "qna_bot.mcp.security.socket.getaddrinfo",
            return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))],
        ):
            with pytest.raises(RegistryRateLimitError):
                official.list_servers()
            assert attempts["count"] == 2

            # Aggregator degrades gracefully with warning rather than crashing Marketplace
            glama = GlamaRegistryProvider(http_client=http, api_key="")
            svc = MCPRegistryService(db_manager=db, official_provider=official, glama_provider=glama, http_client=http)
            page = svc.discover(source="all")
            assert page.items == []
            assert "429" in page.warning
            assert "GLAMA_API_KEY" in page.warning
    finally:
        db.close()


def test_marketplace_stdio_install_requires_confirmation_and_blocks_unsafe_metadata(tmp_path: Path) -> None:
    db = DatabaseManager(db_path=tmp_path / "mkt_stdio.db")
    mgr = MCPClientManager(db_manager=db)
    reg_svc = MCPRegistryService(db_manager=db)
    mcp_svc = MCPService(client_manager=mgr, db_manager=db, registry_service=reg_svc)
    try:
        provider = OfficialMCPRegistryProvider(http_client=reg_svc.http)
        listing = provider._normalize_entry(_fake_official_payload()["servers"][1])
        assert listing is not None
        opt = listing.transport_options[0]

        # Must refuse launching local STDIO process without explicit confirmation flag
        with pytest.raises(MCPSecurityError, match="Explicit confirmation"):
            mcp_svc.install_from_listing(
                user_id="alice",
                listing=listing,
                transport_option=opt,
                stdio_confirmed=False,
            )

        # Even if approved, registration validates allowlist & encrypts secret env vars
        rec, _ = mcp_svc.install_from_listing(
            user_id="alice",
            listing=listing,
            transport_option=opt,
            env_secrets={"MY_SECRET_TOKEN": "super_secret_value"},
            auto_connect=False,
            stdio_confirmed=True,
        )
        assert rec.registry_source == "official_mcp"
        assert rec.command == "uvx"
        # Secret env is NEVER stored plaintext on MCPServerRecord
        assert "super_secret_value" not in str(rec)
        assert mgr.vault.load_env_secrets("alice", rec.server_id) == {"MY_SECRET_TOKEN": "super_secret_value"}

        # Unsafe package identifier injection is rejected before install
        with pytest.raises(MCPSecurityError):
            mcp_svc.install_stdio_package("npm", "badpkg; rm -rf /", confirmed=True)
    finally:
        mgr.shutdown()
        db.close()
