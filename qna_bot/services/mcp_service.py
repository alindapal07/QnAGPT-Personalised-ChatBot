"""Service layer wrapping MCPClientManager, Registry Marketplace, Credential Vault, OAuth, and Audit Logs."""

import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

from qna_bot.config.settings import AppSettings, get_settings
from qna_bot.database.connection import DatabaseManager
from qna_bot.database.models import MCPServerRecord
from qna_bot.database.repositories import AuditLogRepository
from qna_bot.mcp.client_manager import MCPClientManager
from qna_bot.mcp.exceptions import MCPSecurityError
from qna_bot.mcp.models import (
    DiscoveredPrompt,
    DiscoveredResource,
    DiscoveredTool,
    MCPServerStatus,
    PermissionCategory,
)
from qna_bot.mcp.permissions import infer_permission_category, resolve_tool_policy
from qna_bot.mcp.registry.models import RegistryListing, RegistryPage, TransportOption
from qna_bot.mcp.registry.service import MCPRegistryService
from qna_bot.mcp.security import mask_secrets, validate_stdio_config
from qna_bot.mcp.tool_registry import make_namespaced_tool_name

_SAFE_PKG_IDENT_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9@/._\-:=]*$")


class MCPService:
    """High-level backend facade for MCP server lifecycle, marketplace discovery, auth, and logs."""

    def __init__(
        self,
        client_manager: Optional[MCPClientManager] = None,
        db_manager: Optional[DatabaseManager] = None,
        registry_service: Optional[MCPRegistryService] = None,
        settings: Optional[AppSettings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.db = db_manager or (client_manager.db if client_manager else DatabaseManager())
        self.client_manager = client_manager or MCPClientManager(
            db_manager=self.db,
            connect_timeout=self.settings.mcp_connect_timeout_sec,
            tool_timeout=self.settings.mcp_tool_timeout_sec,
        )
        self.registry_service = registry_service or MCPRegistryService(
            db_manager=self.db,
            settings=self.settings,
        )
        self.audit_repo = AuditLogRepository(self.db)

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
        return self.client_manager.register_server(
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
        return self.client_manager.remove_server(server_id=server_id, user_id=user_id)

    def list_servers(self, user_id: str) -> List[MCPServerRecord]:
        return self.client_manager.server_registry.list_servers(user_id=user_id)

    def list_statuses(self, user_id: str) -> List[MCPServerStatus]:
        return self.client_manager.get_all_server_statuses(user_id=user_id)

    def connect(self, server_id: str, user_id: str) -> MCPServerStatus:
        return self.client_manager.connect_server(server_id=server_id, user_id=user_id)

    def disconnect(self, server_id: str, user_id: str) -> None:
        self.client_manager.disconnect_server(server_id=server_id, user_id=user_id)

    def reconnect(self, server_id: str, user_id: str) -> MCPServerStatus:
        return self.client_manager.reconnect_server(server_id=server_id, user_id=user_id)

    def test_connection(self, server_id: str, user_id: str) -> MCPServerStatus:
        return self.client_manager.test_connection(server_id=server_id, user_id=user_id)

    def refresh(self, server_id: str, user_id: str) -> Dict[str, int]:
        return self.client_manager.refresh_server_capabilities(server_id=server_id, user_id=user_id)

    def list_tools(
        self,
        user_id: str,
        server_id: Optional[str] = None,
        include_disabled: bool = False,
    ) -> List[DiscoveredTool]:
        return self.client_manager.list_available_tools(
            user_id=user_id,
            server_id=server_id,
            include_disabled=include_disabled,
        )

    def list_tools_with_persisted(
        self,
        user_id: str,
        server_id: str,
    ) -> List[DiscoveredTool]:
        """Return live discovered tools if connected, or persisted SQLite tool metadata when offline."""
        live = self.client_manager.list_available_tools(
            user_id=user_id,
            server_id=server_id,
            include_disabled=True,
        )
        if live:
            return live

        record = self.client_manager.server_registry.get(server_id=server_id, user_id=user_id)
        if record is None or not record.discovered_tools:
            return []

        result: List[DiscoveredTool] = []
        for meta in record.discovered_tools:
            orig_name = str(meta.get("original_name") or "").strip()
            if not orig_name:
                continue
            override = self.client_manager.permission_repo.get_policy(user_id, server_id, orig_name)
            cat = infer_permission_category(orig_name)
            base = DiscoveredTool(
                namespaced_name=str(meta.get("namespaced_name") or make_namespaced_tool_name(record.name, orig_name)),
                original_name=orig_name,
                server_id=server_id,
                server_name=record.name,
                user_id=user_id,
                description=str(meta.get("description") or ""),
                input_schema=dict(meta.get("input_schema") or {}),
                permission_category=cat,
                auto_approve=(cat == "READ_ONLY"),
            )
            eff_cat, requires_approval = resolve_tool_policy(base, override)
            enabled = bool(override.enabled) if override is not None else bool(meta.get("enabled", True))
            result.append(
                DiscoveredTool(
                    namespaced_name=base.namespaced_name,
                    original_name=base.original_name,
                    server_id=base.server_id,
                    server_name=base.server_name,
                    user_id=base.user_id,
                    description=base.description,
                    input_schema=base.input_schema,
                    permission_category=eff_cat,
                    auto_approve=not requires_approval,
                    enabled=enabled,
                )
            )
        return result

    def list_resources(
        self,
        user_id: str,
        server_id: Optional[str] = None,
    ) -> List[DiscoveredResource]:
        return self.client_manager.list_available_resources(user_id=user_id, server_id=server_id)

    def list_prompts(
        self,
        user_id: str,
        server_id: Optional[str] = None,
    ) -> List[DiscoveredPrompt]:
        return self.client_manager.list_available_prompts(user_id=user_id, server_id=server_id)

    def read_resource(self, user_id: str, server_id: str, uri: str) -> Dict[str, str]:
        return self.client_manager.read_resource(user_id=user_id, server_id=server_id, uri=uri)

    def get_prompt(
        self,
        user_id: str,
        server_id: str,
        name: str,
        arguments: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        return self.client_manager.get_prompt(
            user_id=user_id,
            server_id=server_id,
            name=name,
            arguments=arguments,
        )

    def set_tool_enabled(
        self,
        user_id: str,
        server_id: str,
        tool_name: str,
        enabled: bool,
    ) -> Optional[DiscoveredTool]:
        return self.client_manager.set_tool_enabled(
            user_id=user_id,
            server_id=server_id,
            tool_name=tool_name,
            enabled=enabled,
        )

    def check_health(self, server_id: str, user_id: str) -> MCPServerStatus:
        return self.client_manager.check_health(server_id=server_id, user_id=user_id)

    def set_tool_permission(
        self,
        user_id: str,
        server_id: str,
        tool_name: str,
        category: PermissionCategory,
        auto_approve: bool,
        enabled: Optional[bool] = None,
    ) -> None:
        self.client_manager.set_tool_permission(
            user_id=user_id,
            server_id=server_id,
            tool_name=tool_name,
            category=category,
            auto_approve=auto_approve,
            enabled=enabled,
        )

    # ── Credential Vault & OAuth Delegates ───────────────────────────────────
    def store_bearer_token(
        self,
        user_id: str,
        server_id: str,
        token: str,
        header_name: str = "Authorization",
    ) -> None:
        self.client_manager.store_bearer_token(user_id, server_id, token, header_name=header_name)

    def store_api_key(
        self,
        user_id: str,
        server_id: str,
        api_key: str,
        header_name: str = "X-API-Key",
    ) -> None:
        self.client_manager.store_api_key(user_id, server_id, api_key, header_name=header_name)

    def store_env_secrets(self, user_id: str, server_id: str, secrets_map: Dict[str, str]) -> None:
        self.client_manager.store_env_secrets(user_id, server_id, secrets_map)

    def get_credential_status(self, user_id: str, server_id: str) -> Dict[str, Any]:
        return self.client_manager.get_credential_status(user_id, server_id)

    def start_oauth_flow(self, user_id: str, server_id: str, **kwargs: Any):
        return self.client_manager.start_oauth_flow(user_id=user_id, server_id=server_id, **kwargs)

    def complete_oauth_flow(
        self,
        user_id: str,
        server_id: str,
        callback_url_or_code: str,
        state: Optional[str] = None,
    ):
        return self.client_manager.complete_oauth_flow(
            user_id=user_id,
            server_id=server_id,
            callback_url_or_code=callback_url_or_code,
            state=state,
        )

    def refresh_oauth_token(self, user_id: str, server_id: str):
        return self.client_manager.refresh_oauth_token(user_id=user_id, server_id=server_id)

    def revoke_credentials(self, user_id: str, server_id: str) -> bool:
        return self.client_manager.revoke_credentials(user_id=user_id, server_id=server_id)

    # ── MCP Marketplace Registry & Controlled Installation ───────────────────
    def discover_marketplace(
        self,
        source: str = "all",
        query: str = "",
        category: str = "All",
        feed: str = "all",
        transport_filter: str = "All",
        only_connectable: bool = False,
        limit: int = 24,
        cursor: Optional[str] = None,
    ) -> RegistryPage:
        return self.registry_service.discover(
            source=source,
            query=query,
            category=category,
            feed=feed,
            transport_filter=transport_filter,
            only_connectable=only_connectable,
            limit=limit,
            cursor=cursor,
        )

    def get_marketplace_details(self, registry_source: str, registry_id: str) -> Optional[RegistryListing]:
        return self.registry_service.get_details(registry_source, registry_id)

    def clear_registry_cache(self, source: Optional[str] = None) -> None:
        self.registry_service.clear_cache(source=source)

    def map_installed_connectors(self, user_id: str) -> Dict[str, tuple[MCPServerRecord, MCPServerStatus]]:
        """Map `registry_id` (and `registry_source::registry_id`) -> (record, status) for UI badges."""
        records = self.list_servers(user_id)
        statuses = {s.server_id: s for s in self.list_statuses(user_id)}
        mapping: Dict[str, tuple[MCPServerRecord, MCPServerStatus]] = {}
        for rec in records:
            st = statuses.get(rec.server_id)
            if st is None:
                continue
            if rec.registry_id:
                mapping[f"{rec.registry_source}::{rec.registry_id}"] = (rec, st)
                mapping[rec.registry_id] = (rec, st)
            mapping[rec.name.lower()] = (rec, st)
        return mapping

    def check_stdio_runtime(self, command: str) -> tuple[bool, str]:
        """Verify an allowlisted local runtime executable exists on PATH."""
        cmd = (command or "").strip()
        if not cmd:
            return False, "No command specified."
        validate_stdio_config(command=cmd, args=[])
        resolved = shutil.which(cmd)
        if resolved:
            return True, f"Runtime `{cmd}` available at `{resolved}`."
        return False, f"Runtime `{cmd}` was not found on PATH."

    def install_stdio_package(
        self,
        package_type: str,
        identifier: str,
        version: str = "",
        confirmed: bool = False,
    ) -> tuple[bool, str]:
        """Perform controlled, explicit-user-approved package installation without shell invocation."""
        if not confirmed:
            raise MCPSecurityError(
                "Explicit user approval is required before installing local packages."
            )
        pkg_id = (identifier or "").strip()
        if not pkg_id or not _SAFE_PKG_IDENT_RE.match(pkg_id):
            raise MCPSecurityError("Package identifier failed security format validation.")

        pkg_kind = (package_type or "").strip().lower()
        target = f"{pkg_id}@{version.strip()}" if version.strip() and pkg_kind == "npm" else (
            f"{pkg_id}=={version.strip()}" if version.strip() and pkg_kind == "pypi" else pkg_id
        )
        if pkg_kind == "pypi":
            uv_bin = shutil.which("uv")
            cmd = [uv_bin, "tool", "install", target] if uv_bin else [sys.executable, "-m", "pip", "install", target]
        elif pkg_kind == "npm":
            npm_bin = shutil.which("npm")
            if not npm_bin:
                return False, "`npm` executable not found on PATH."
            cmd = [npm_bin, "install", "--no-audit", "--no-fund", target]
        elif pkg_kind in ("oci", "docker"):
            docker_bin = shutil.which("docker")
            if not docker_bin:
                return False, "`docker` executable not found on PATH."
            cmd = [docker_bin, "pull", target]
        else:
            raise MCPSecurityError(f"Unsupported automated install type '{package_type}'.")

        _, cleaned_args, safe_env = validate_stdio_config(command="uv", args=cmd[1:])
        proc = subprocess.run(
            [cmd[0], *cleaned_args],
            env=safe_env,
            capture_output=True,
            text=True,
            timeout=120,
            shell=False,
            check=False,
        )
        out = str(mask_secrets((proc.stdout or "") + "\n" + (proc.stderr or ""))).strip()[:1200]
        return proc.returncode == 0, out

    def install_from_listing(
        self,
        user_id: str,
        listing: RegistryListing,
        transport_option: TransportOption,
        auth_type: str = "none",
        bearer_token: str = "",
        api_key: str = "",
        auth_header_name: str = "Authorization",
        auth_env_var: str = "",
        env_secrets: Optional[Mapping[str, str]] = None,
        env_refs: Optional[Dict[str, str]] = None,
        allowed_roots: Optional[Sequence[str]] = None,
        allow_insecure_http: bool = False,
        allow_private_network: bool = False,
        auto_connect: bool = True,
        stdio_confirmed: bool = False,
    ) -> tuple[MCPServerRecord, Optional[MCPServerStatus]]:
        """Register & securely authenticate a Marketplace connector, then connect when ready."""
        if not transport_option.direct_connectable:
            raise MCPSecurityError(
                transport_option.notes
                or "Selected connector option cannot be connected automatically."
            )
        if transport_option.transport == "stdio" and not stdio_confirmed:
            raise MCPSecurityError(
                "Explicit confirmation is required before launching a local STDIO MCP server."
            )

        rec = self.register_server(
            user_id=user_id,
            name=listing.title or listing.name,
            transport=transport_option.transport,
            enabled=True,
            command=transport_option.command,
            args=list(transport_option.args),
            env_refs=env_refs,
            url=transport_option.url,
            auth_type=auth_type,
            auth_header_name=auth_header_name,
            auth_env_var=auth_env_var,
            allow_insecure_http=allow_insecure_http,
            allow_private_network=allow_private_network,
            allowed_roots=allowed_roots,
            registry_source=listing.registry_source,
            registry_id=listing.registry_id,
            listing_url=listing.listing_url,
            description=listing.description,
            category=listing.category,
            icon_url=listing.icon_url,
            package_type=transport_option.package_type,
            install_hint=transport_option.install_hint,
        )

        if auth_type == "bearer_token" and bearer_token.strip():
            self.store_bearer_token(user_id, rec.server_id, bearer_token, header_name=auth_header_name)
        elif auth_type == "api_key" and api_key.strip():
            self.store_api_key(user_id, rec.server_id, api_key, header_name=auth_header_name or "X-API-Key")

        if env_secrets:
            self.store_env_secrets(user_id, rec.server_id, dict(env_secrets))

        status: Optional[MCPServerStatus] = None
        if auto_connect and auth_type != "oauth":
            status = self.connect(rec.server_id, user_id)
        return rec, status

    def list_audit_logs(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        return self.audit_repo.list_recent(user_id=user_id, limit=limit)
