"""Central MCP Client Manager handling multi-server sessions, discovery, health, and tool execution."""

from contextlib import AsyncExitStack
import threading
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from qna_bot.core.logging_config import get_logger
from qna_bot.database.connection import DatabaseManager
from qna_bot.database.models import MCPServerRecord, ToolPermissionRecord
from qna_bot.database.repositories import MCPServerRepository, PermissionRepository
from qna_bot.mcp.config import DEFAULT_CONNECT_TIMEOUT_SEC, DEFAULT_TOOL_TIMEOUT_SEC
from qna_bot.mcp.connection import AsyncLoopRunner, MCPConnectionHandle
from qna_bot.mcp.credentials import CredentialVault
from qna_bot.mcp.exceptions import MCPConnectionError, MCPError, MCPToolExecutionError
from qna_bot.mcp.models import (
    DiscoveredPrompt,
    DiscoveredResource,
    DiscoveredTool,
    MCPServerStatus,
    PermissionCategory,
    ToolExecutionResult,
)
from qna_bot.mcp.oauth import OAuthFlowManager
from qna_bot.mcp.prompt_manager import MCPPromptManager
from qna_bot.mcp.resource_manager import MCPResourceManager
from qna_bot.mcp.security import sanitize_tool_output
from qna_bot.mcp.server_registry import MCPServerRegistry
from qna_bot.mcp.tool_registry import MCPToolRegistry

logger = get_logger(__name__)


def _extract_tool_output_text(raw_result: Any) -> tuple[str, Any, bool]:
    """Convert MCP CallToolResult content blocks into sanitized text + structured data."""
    is_error = bool(getattr(raw_result, "is_error", False) or getattr(raw_result, "isError", False))
    structured = getattr(raw_result, "structured_content", None)
    if structured is None:
        structured = getattr(raw_result, "structuredContent", None)

    content_blocks = getattr(raw_result, "content", None) or []
    parts: List[str] = []
    for block in content_blocks:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(str(text))
        else:
            data = getattr(block, "data", None)
            mime = getattr(block, "mime_type", None) or getattr(block, "mimeType", None) or "blob"
            if data is not None:
                parts.append(f"[{mime} payload: {len(str(data))} chars]")
            else:
                parts.append(str(block))

    if not parts and structured is not None:
        import json

        parts.append(json.dumps(structured, indent=2, default=str))

    return sanitize_tool_output("\n".join(parts)), structured, is_error


class MCPClientManager:
    """Host-side MCP client orchestrating per-user connections, capabilities, and execution."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_SEC,
        tool_timeout: float = DEFAULT_TOOL_TIMEOUT_SEC,
        session_factory: Optional[Callable[[MCPServerRecord, AsyncExitStack], Awaitable[Any]]] = None,
    ) -> None:
        self.db = db_manager or DatabaseManager()
        self.server_repo = MCPServerRepository(self.db)
        self.permission_repo = PermissionRepository(self.db)
        self.server_registry = MCPServerRegistry(self.server_repo)
        self.tool_registry = MCPToolRegistry(permission_repo=self.permission_repo)
        self.resource_manager = MCPResourceManager()
        self.prompt_manager = MCPPromptManager()
        self.vault = CredentialVault(db_manager=self.db)
        self.oauth_manager = OAuthFlowManager(vault=self.vault)
        self._loop_runner = AsyncLoopRunner()
        self._connect_timeout = connect_timeout
        self._tool_timeout = tool_timeout
        self._session_factory = session_factory
        self._lock = threading.RLock()
        self._connections: Dict[tuple[str, str], MCPConnectionHandle] = {}

    def shutdown(self) -> None:
        """Disconnect all active MCP sessions and stop the background event loop."""
        self.disconnect_all_servers()
        self._loop_runner.close()

    def register_server(
        self,
        user_id: str,
        name: str,
        transport: str,
        enabled: bool = True,
        command: str = "",
        args: Optional[Sequence[str]] = None,
        env_refs: Optional[Dict[str, str]] = None,
        url: str = "",
        auth_type: str = "none",
        auth_header_name: str = "Authorization",
        auth_env_var: str = "",
        allow_insecure_http: bool = False,
        allow_private_network: bool = False,
        allowed_roots: Optional[Sequence[str]] = None,
        server_id: Optional[str] = None,
        registry_source: str = "manual",
        registry_id: str = "",
        listing_url: str = "",
        description: str = "",
        category: str = "General",
        icon_url: str = "",
        package_type: str = "",
        install_hint: str = "",
    ) -> MCPServerRecord:
        return self.server_registry.register(
            user_id=user_id,
            name=name,
            transport=transport,
            enabled=enabled,
            command=command,
            args=args,
            env_refs=env_refs,
            url=url,
            auth_type=auth_type,
            auth_header_name=auth_header_name,
            auth_env_var=auth_env_var,
            allow_insecure_http=allow_insecure_http,
            allow_private_network=allow_private_network,
            allowed_roots=allowed_roots,
            server_id=server_id,
            registry_source=registry_source,
            registry_id=registry_id,
            listing_url=listing_url,
            description=description,
            category=category,
            icon_url=icon_url,
            package_type=package_type,
            install_hint=install_hint,
        )

    def remove_server(self, server_id: str, user_id: str) -> bool:
        self.disconnect_server(server_id=server_id, user_id=user_id)
        return self.server_registry.remove(server_id=server_id, user_id=user_id)

    def _require_server(self, server_id: str, user_id: str) -> MCPServerRecord:
        record = self.server_registry.get(server_id=server_id, user_id=user_id)
        if record is None:
            raise MCPError(f"MCP server '{server_id}' not found for user '{user_id}'.")
        return record

    def connect_server(self, server_id: str, user_id: str) -> MCPServerStatus:
        record = self._require_server(server_id, user_id)
        key = (user_id, server_id)

        with self._lock:
            existing = self._connections.get(key)
            if existing is not None and existing.is_connected:
                return self.get_server_status(server_id, user_id)

            handle = MCPConnectionHandle(
                record=record,
                loop_runner=self._loop_runner,
                connect_timeout=self._connect_timeout,
                session_factory=self._session_factory,
            )
            if hasattr(handle, "configure_auth"):
                handle.configure_auth(self.vault, self.oauth_manager)
            if hasattr(handle, "configure_timeouts"):
                handle.configure_timeouts(self._tool_timeout)
            try:
                handle.connect()
                self._connections[key] = handle
                self.server_repo.update_status(
                    server_id=server_id,
                    user_id=user_id,
                    status="connected",
                    last_error="",
                    last_connected_at=time.time(),
                )
                self.server_repo.update_health(
                    server_id=server_id,
                    user_id=user_id,
                    health_status="healthy",
                )
                self.refresh_server_capabilities(server_id=server_id, user_id=user_id)
            except Exception as exc:
                handle.disconnect()
                self._connections.pop(key, None)
                self.server_repo.update_status(
                    server_id=server_id,
                    user_id=user_id,
                    status="error",
                    last_error=str(exc),
                )
                self.server_repo.update_health(
                    server_id=server_id,
                    user_id=user_id,
                    health_status="unhealthy",
                )
                raise

        return self.get_server_status(server_id, user_id)

    def disconnect_server(self, server_id: str, user_id: str) -> None:
        key = (user_id, server_id)
        with self._lock:
            handle = self._connections.pop(key, None)
            if handle is not None:
                handle.disconnect()
            self.tool_registry.clear_server_tools(user_id, server_id)
            self.resource_manager.clear_server_resources(user_id, server_id)
            self.prompt_manager.clear_server_prompts(user_id, server_id)
            if self.server_registry.get(server_id, user_id) is not None:
                self.server_repo.update_status(
                    server_id=server_id,
                    user_id=user_id,
                    status="disconnected",
                    last_error="",
                )

    def reconnect_server(self, server_id: str, user_id: str) -> MCPServerStatus:
        self.disconnect_server(server_id, user_id)
        return self.connect_server(server_id, user_id)

    def connect_all_enabled_servers(self, user_id: str) -> List[MCPServerStatus]:
        statuses: List[MCPServerStatus] = []
        for record in self.server_registry.list_servers(user_id):
            if not record.enabled:
                statuses.append(self.get_server_status(record.server_id, user_id))
                continue
            try:
                statuses.append(self.connect_server(record.server_id, user_id))
            except Exception as exc:
                logger.warning("Auto-connect failed for '%s': %s", record.name, exc)
                statuses.append(self.get_server_status(record.server_id, user_id))
        return statuses

    def disconnect_all_servers(self, user_id: Optional[str] = None) -> None:
        with self._lock:
            keys = [
                k for k in list(self._connections.keys()) if user_id is None or k[0] == user_id
            ]
            for uid, sid in keys:
                self.disconnect_server(server_id=sid, user_id=uid)

    def refresh_server_capabilities(self, server_id: str, user_id: str) -> Dict[str, int]:
        record = self._require_server(server_id, user_id)
        with self._lock:
            handle = self._connections.get((user_id, server_id))
            if handle is None or not handle.is_connected:
                raise MCPConnectionError(f"Server '{record.name}' is not connected.")

            async def _discover(session: Any) -> tuple[list, list, list]:
                tools_res = await session.list_tools()
                tools_list = list(getattr(tools_res, "tools", None) or [])

                resources_list: list = []
                try:
                    res_out = await session.list_resources()
                    resources_list = list(getattr(res_out, "resources", None) or [])
                except Exception:
                    resources_list = []

                prompts_list: list = []
                try:
                    prm_out = await session.list_prompts()
                    prompts_list = list(getattr(prm_out, "prompts", None) or [])
                except Exception:
                    prompts_list = []

                return tools_list, resources_list, prompts_list

            try:
                raw_tools, raw_resources, raw_prompts = handle.run_call(
                    _discover,
                    timeout=self._connect_timeout,
                )
            except Exception as exc:
                raise MCPConnectionError(
                    f"Capability discovery failed for '{record.name}': {exc}"
                ) from exc

            t = self.tool_registry.register_server_tools(
                user_id, server_id, record.name, raw_tools
            )
            r = self.resource_manager.register_server_resources(
                user_id, server_id, record.name, raw_resources
            )
            p = self.prompt_manager.register_server_prompts(
                user_id, server_id, record.name, raw_prompts
            )
            # Persist normalized discovered tool metadata in SQLite so tools remain inspectable offline
            persisted_meta = [
                {
                    "namespaced_name": item.namespaced_name,
                    "original_name": item.original_name,
                    "description": item.description,
                    "input_schema": item.input_schema,
                    "permission_category": item.permission_category,
                    "auto_approve": item.auto_approve,
                    "enabled": item.enabled,
                }
                for item in t
            ]
            self.server_repo.update_discovered_tools(server_id, user_id, persisted_meta)
            return {"tools": len(t), "resources": len(r), "prompts": len(p)}

    def check_health(self, server_id: str, user_id: str) -> MCPServerStatus:
        """Actively ping a connected MCP server and persist health telemetry."""
        record = self._require_server(server_id, user_id)
        key = (user_id, server_id)
        handle = self._connections.get(key)
        if handle is None or not handle.is_connected:
            self.server_repo.update_health(server_id, user_id, "disconnected")
            return self.get_server_status(server_id, user_id)
        try:
            handle.run_call(lambda sess: sess.send_ping(), timeout=5.0)
            self.server_repo.update_health(server_id, user_id, "healthy")
        except Exception as exc:
            logger.warning("Health check failed for MCP server '%s': %s", record.name, exc)
            self.server_repo.update_health(server_id, user_id, "degraded")
            self.server_repo.update_status(
                server_id=server_id,
                user_id=user_id,
                status="error",
                last_error=f"Health check failed: {exc}",
            )
        return self.get_server_status(server_id, user_id)

    def test_connection(self, server_id: str, user_id: str) -> MCPServerStatus:
        """Perform handshake + capability discovery; verifies ping health cleanly."""
        key = (user_id, server_id)
        was_connected = key in self._connections and self._connections[key].is_connected
        status = self.connect_server(server_id, user_id)
        if not was_connected:
            status = self.check_health(server_id, user_id)
        return status

    def get_server_status(self, server_id: str, user_id: str) -> MCPServerStatus:
        record = self._require_server(server_id, user_id)
        handle = self._connections.get((user_id, server_id))
        is_conn = bool(handle is not None and handle.is_connected)
        tool_cnt = len(self.tool_registry.list_tools(user_id, server_id=server_id, include_disabled=True))
        if not is_conn and tool_cnt == 0 and record.discovered_tools:
            tool_cnt = len(record.discovered_tools)
        res_cnt = len(self.resource_manager.list_resources(user_id, server_id=server_id))
        prm_cnt = len(self.prompt_manager.list_prompts(user_id, server_id=server_id))
        cred_status = self.vault.get_credential_status(user_id=user_id, server_id=server_id)
        return MCPServerStatus(
            server_id=record.server_id,
            name=record.name,
            transport=record.transport,
            enabled=record.enabled,
            connected=is_conn,
            status="connected" if is_conn else record.status,
            tool_count=tool_cnt,
            resource_count=res_cnt,
            prompt_count=prm_cnt,
            last_connected_at=record.last_connected_at,
            last_error=record.last_error,
            registry_source=record.registry_source,
            listing_url=record.listing_url,
            auth_type=record.auth_type,
            credential_configured=bool(cred_status.get("configured")) or record.auth_type == "none",
            credential_expired=bool(cred_status.get("expired")),
            health_status="healthy" if (is_conn and record.last_health_status == "unknown") else record.last_health_status,
            health_checked_at=record.health_checked_at,
        )

    def get_all_server_statuses(self, user_id: str) -> List[MCPServerStatus]:
        return [
            self.get_server_status(r.server_id, user_id)
            for r in self.server_registry.list_servers(user_id)
        ]

    def list_available_tools(
        self,
        user_id: str,
        server_id: Optional[str] = None,
        include_disabled: bool = False,
    ) -> List[DiscoveredTool]:
        return self.tool_registry.list_tools(
            user_id=user_id,
            server_id=server_id,
            include_disabled=include_disabled,
        )

    def set_tool_enabled(
        self,
        user_id: str,
        server_id: str,
        tool_name: str,
        enabled: bool,
    ) -> Optional[DiscoveredTool]:
        self._require_server(server_id, user_id)
        return self.tool_registry.set_tool_enabled(
            user_id=user_id,
            server_id=server_id,
            tool_name=tool_name,
            enabled=enabled,
        )

    # ── Credential & OAuth Lifecycle Helpers ─────────────────────────────────
    def store_bearer_token(
        self,
        user_id: str,
        server_id: str,
        token: str,
        header_name: str = "Authorization",
    ) -> None:
        self._require_server(server_id, user_id)
        self.vault.store_bearer_token(user_id=user_id, server_id=server_id, token=token, header_name=header_name)

    def store_api_key(
        self,
        user_id: str,
        server_id: str,
        api_key: str,
        header_name: str = "X-API-Key",
    ) -> None:
        self._require_server(server_id, user_id)
        self.vault.store_api_key(user_id=user_id, server_id=server_id, api_key=api_key, header_name=header_name)

    def store_env_secrets(self, user_id: str, server_id: str, secrets_map: Dict[str, str]) -> None:
        self._require_server(server_id, user_id)
        self.vault.store_env_secrets(user_id=user_id, server_id=server_id, secrets_map=secrets_map)

    def get_credential_status(self, user_id: str, server_id: str) -> Dict[str, Any]:
        self._require_server(server_id, user_id)
        return self.vault.get_credential_status(user_id=user_id, server_id=server_id)

    def start_oauth_flow(
        self,
        user_id: str,
        server_id: str,
        client_id: str = "",
        client_secret: str = "",
        authorization_endpoint: str = "",
        token_endpoint: str = "",
        revocation_endpoint: str = "",
        redirect_uri: str = "",
        scopes: str = "",
    ):
        record = self._require_server(server_id, user_id)
        return self.oauth_manager.start_authorization_flow(
            user_id=user_id,
            server_id=server_id,
            server_url=record.url,
            client_id=client_id,
            client_secret=client_secret,
            authorization_endpoint=authorization_endpoint,
            token_endpoint=token_endpoint,
            revocation_endpoint=revocation_endpoint,
            redirect_uri=redirect_uri,
            scopes=scopes,
            allow_insecure_http=record.allow_insecure_http,
            allow_private_network=record.allow_private_network,
        )

    def complete_oauth_flow(
        self,
        user_id: str,
        server_id: str,
        callback_url_or_code: str,
        state: Optional[str] = None,
    ):
        self._require_server(server_id, user_id)
        return self.oauth_manager.complete_authorization_callback(
            user_id=user_id,
            server_id=server_id,
            callback_url_or_code=callback_url_or_code,
            state=state,
        )

    def refresh_oauth_token(self, user_id: str, server_id: str):
        record = self._require_server(server_id, user_id)
        return self.oauth_manager.refresh_access_token(
            user_id=user_id,
            server_id=server_id,
            allow_insecure_http=record.allow_insecure_http,
            allow_private_network=record.allow_private_network,
        )

    def revoke_credentials(self, user_id: str, server_id: str) -> bool:
        record = self._require_server(server_id, user_id)
        self.disconnect_server(server_id=server_id, user_id=user_id)
        return self.oauth_manager.revoke_access(
            user_id=user_id,
            server_id=server_id,
            allow_insecure_http=record.allow_insecure_http,
            allow_private_network=record.allow_private_network,
        )

    def list_available_resources(
        self,
        user_id: str,
        server_id: Optional[str] = None,
    ) -> List[DiscoveredResource]:
        return self.resource_manager.list_resources(user_id=user_id, server_id=server_id)

    def list_available_prompts(
        self,
        user_id: str,
        server_id: Optional[str] = None,
    ) -> List[DiscoveredPrompt]:
        return self.prompt_manager.list_prompts(user_id=user_id, server_id=server_id)

    def set_tool_permission(
        self,
        user_id: str,
        server_id: str,
        tool_name: str,
        category: PermissionCategory,
        auto_approve: bool,
        enabled: Optional[bool] = None,
    ) -> None:
        self._require_server(server_id, user_id)
        existing = self.permission_repo.get_policy(user_id, server_id, tool_name)
        is_enabled = enabled if enabled is not None else (existing.enabled if existing is not None else True)
        self.permission_repo.set_policy(
            ToolPermissionRecord(
                user_id=user_id,
                server_id=server_id,
                tool_name=tool_name,
                category=category,
                auto_approve=auto_approve,
                enabled=bool(is_enabled),
                updated_at=time.time(),
            )
        )
        if (user_id, server_id) in self._connections:
            self.refresh_server_capabilities(server_id, user_id)

    def execute_tool(
        self,
        user_id: str,
        namespaced_tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> ToolExecutionResult:
        tool = self.tool_registry.get_tool(user_id=user_id, namespaced_name=namespaced_tool_name)
        if tool is None:
            raise MCPToolExecutionError(
                f"Tool '{namespaced_tool_name}' is not available for user '{user_id}'."
            )

        handle = self._connections.get((user_id, tool.server_id))
        if handle is None or not handle.is_connected:
            raise MCPConnectionError(
                f"MCP server '{tool.server_name}' is not connected."
            )

        t0 = time.perf_counter()
        call_timeout = timeout or self._tool_timeout
        args = dict(arguments or {})

        try:
            raw_result = handle.run_call(
                lambda sess: sess.call_tool(tool.original_name, arguments=args),
                timeout=call_timeout,
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            text, structured, is_error = _extract_tool_output_text(raw_result)
            return ToolExecutionResult(
                server_id=tool.server_id,
                server_name=tool.server_name,
                tool_name=tool.original_name,
                success=not is_error,
                content=text,
                structured_content=structured,
                is_error=is_error,
                duration_ms=elapsed_ms,
                error=text if is_error else "",
            )
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            logger.warning(
                "MCP tool '%s' on '%s' failed: %s",
                tool.original_name,
                tool.server_name,
                exc,
            )
            raise MCPToolExecutionError(
                f"MCP tool '{tool.original_name}' failed: {exc}"
            ) from exc

    def read_resource(self, user_id: str, server_id: str, uri: str) -> Dict[str, str]:
        record = self._require_server(server_id, user_id)
        handle = self._connections.get((user_id, server_id))
        if handle is None or not handle.is_connected:
            raise MCPConnectionError(f"MCP server '{record.name}' is not connected.")

        try:
            raw = handle.run_call(
                lambda sess: sess.read_resource(uri),
                timeout=self._tool_timeout,
            )
            return self.resource_manager.format_read_result(raw)
        except Exception as exc:
            raise MCPConnectionError(f"Failed reading resource '{uri}': {exc}") from exc

    def get_prompt(
        self,
        user_id: str,
        server_id: str,
        name: str,
        arguments: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        record = self._require_server(server_id, user_id)
        handle = self._connections.get((user_id, server_id))
        if handle is None or not handle.is_connected:
            raise MCPConnectionError(f"MCP server '{record.name}' is not connected.")

        try:
            raw = handle.run_call(
                lambda sess: sess.get_prompt(name, arguments=arguments or {}),
                timeout=self._tool_timeout,
            )
            return self.prompt_manager.format_prompt_result(raw)
        except Exception as exc:
            raise MCPConnectionError(f"Failed fetching prompt '{name}': {exc}") from exc
