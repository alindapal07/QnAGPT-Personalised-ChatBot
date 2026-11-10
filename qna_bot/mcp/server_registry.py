"""Dynamic per-user MCP server configuration registry backed by SQLite."""

import re
from typing import Dict, List, Optional, Sequence
import uuid

from qna_bot.database.models import MCPServerRecord
from qna_bot.database.repositories import MCPServerRepository
from qna_bot.mcp.exceptions import MCPError
from qna_bot.mcp.security import validate_remote_url, validate_stdio_config

_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")


def slugify_server_name(name: str) -> str:
    slug = _SLUG_RE.sub("_", (name or "").strip().lower()).strip("_")
    return slug or "mcp_server"


class MCPServerRegistry:
    """Validates and persists user-scoped MCP server configurations."""

    def __init__(self, repo: MCPServerRepository) -> None:
        self.repo = repo

    def validate_config(self, record: MCPServerRecord) -> None:
        if not (record.name or "").strip():
            raise MCPError("Server name is required.")
        if not (record.user_id or "").strip():
            raise MCPError("user_id is required for server registration.")

        transport = (record.transport or "").lower()
        if transport == "stdio":
            validate_stdio_config(
                command=record.command,
                args=record.args,
                env_refs=record.env_refs,
                allowed_roots=record.allowed_roots,
            )
        elif transport in ("streamable_http", "sse"):
            validate_remote_url(
                record.url,
                allow_insecure_http=record.allow_insecure_http,
                allow_private_network=record.allow_private_network,
            )
        else:
            raise MCPError(f"Unsupported transport '{record.transport}'.")

    def register(
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
        sid = (server_id or f"{slugify_server_name(name)}_{uuid.uuid4().hex[:8]}").strip()
        existing = self.get(sid, user_id.strip())
        record = MCPServerRecord(
            server_id=sid,
            user_id=user_id.strip(),
            name=name.strip(),
            transport=transport.strip().lower(),
            enabled=enabled,
            command=command.strip(),
            args=list(args or []),
            env_refs=dict(env_refs or {}),
            url=url.strip(),
            auth_type=auth_type.strip().lower(),
            auth_header_name=auth_header_name.strip() or "Authorization",
            auth_env_var=auth_env_var.strip(),
            allow_insecure_http=allow_insecure_http,
            allow_private_network=allow_private_network,
            allowed_roots=list(allowed_roots or []),
            registry_source=registry_source.strip() or (existing.registry_source if existing else "manual"),
            registry_id=registry_id.strip() or (existing.registry_id if existing else ""),
            listing_url=listing_url.strip() or (existing.listing_url if existing else ""),
            description=description.strip() or (existing.description if existing else ""),
            category=category.strip() or (existing.category if existing else "General"),
            icon_url=icon_url.strip() or (existing.icon_url if existing else ""),
            package_type=package_type.strip() or (existing.package_type if existing else ""),
            install_hint=install_hint.strip() or (existing.install_hint if existing else ""),
            discovered_tools=list(existing.discovered_tools) if existing else [],
            created_at=existing.created_at if existing else 0.0,
        )
        self.validate_config(record)
        return self.repo.upsert(record)

    def get(self, server_id: str, user_id: str) -> Optional[MCPServerRecord]:
        return self.repo.get(server_id=server_id, user_id=user_id)

    def list_servers(self, user_id: str) -> List[MCPServerRecord]:
        return self.repo.list_for_user(user_id=user_id)

    def remove(self, server_id: str, user_id: str) -> bool:
        return self.repo.delete(server_id=server_id, user_id=user_id)
