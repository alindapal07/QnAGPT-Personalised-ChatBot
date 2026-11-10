"""User-isolated SQLite repositories for MCP servers, permissions, approvals, and audit logs."""

import json
import time
from typing import Any, Dict, List, Optional

from qna_bot.core.exceptions import MemoryPersistenceError
from qna_bot.database.connection import DatabaseManager
from qna_bot.database.models import (
    ApprovalRecord,
    AuditLogRecord,
    EncryptedCredentialRecord,
    MCPServerRecord,
    ToolPermissionRecord,
)


class MCPServerRepository:
    """Persists MCP server configurations strictly isolated per user_id."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    @staticmethod
    def _row_to_record(row: Any) -> MCPServerRecord:
        return MCPServerRecord(
            server_id=row["server_id"],
            user_id=row["user_id"],
            name=row["name"],
            transport=row["transport"],
            enabled=bool(row["enabled"]),
            command=row["command"],
            args=json.loads(row["args_json"] or "[]"),
            env_refs=json.loads(row["env_refs_json"] or "{}"),
            url=row["url"],
            auth_type=row["auth_type"],
            auth_header_name=row["auth_header_name"],
            auth_env_var=row["auth_env_var"],
            allow_insecure_http=bool(row["allow_insecure_http"]),
            allow_private_network=bool(row["allow_private_network"]),
            allowed_roots=json.loads(row["allowed_roots_json"] or "[]"),
            status=row["status"],
            last_connected_at=row["last_connected_at"],
            last_error=row["last_error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            registry_source=row["registry_source"] or "manual",
            registry_id=row["registry_id"] or "",
            listing_url=row["listing_url"] or "",
            description=row["description"] or "",
            category=row["category"] or "General",
            icon_url=row["icon_url"] or "",
            package_type=row["package_type"] or "",
            install_hint=row["install_hint"] or "",
            discovered_tools=json.loads(row["discovered_tools_json"] or "[]"),
            last_health_status=row["last_health_status"] or "unknown",
            health_checked_at=row["health_checked_at"],
        )

    def upsert(self, record: MCPServerRecord) -> MCPServerRecord:
        now = time.time()
        created = record.created_at or now
        with self.db.lock:
            existing = self.db.connection.execute(
                "SELECT user_id FROM mcp_servers WHERE server_id = ?",
                (record.server_id,),
            ).fetchone()
            if existing is not None and existing["user_id"] != record.user_id:
                raise MemoryPersistenceError("Unauthorized MCP server ownership mismatch.")

            self.db.connection.execute(
                """
                INSERT INTO mcp_servers (
                    server_id, user_id, name, transport, enabled, command,
                    args_json, env_refs_json, url, auth_type, auth_header_name,
                    auth_env_var, allow_insecure_http, allow_private_network,
                    allowed_roots_json, status, last_connected_at, last_error,
                    created_at, updated_at, registry_source, registry_id,
                    listing_url, description, category, icon_url, package_type,
                    install_hint, discovered_tools_json, last_health_status,
                    health_checked_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(server_id) DO UPDATE SET
                    name=excluded.name,
                    transport=excluded.transport,
                    enabled=excluded.enabled,
                    command=excluded.command,
                    args_json=excluded.args_json,
                    env_refs_json=excluded.env_refs_json,
                    url=excluded.url,
                    auth_type=excluded.auth_type,
                    auth_header_name=excluded.auth_header_name,
                    auth_env_var=excluded.auth_env_var,
                    allow_insecure_http=excluded.allow_insecure_http,
                    allow_private_network=excluded.allow_private_network,
                    allowed_roots_json=excluded.allowed_roots_json,
                    status=excluded.status,
                    last_connected_at=excluded.last_connected_at,
                    last_error=excluded.last_error,
                    updated_at=excluded.updated_at,
                    registry_source=excluded.registry_source,
                    registry_id=excluded.registry_id,
                    listing_url=excluded.listing_url,
                    description=excluded.description,
                    category=excluded.category,
                    icon_url=excluded.icon_url,
                    package_type=excluded.package_type,
                    install_hint=excluded.install_hint,
                    discovered_tools_json=excluded.discovered_tools_json,
                    last_health_status=excluded.last_health_status,
                    health_checked_at=excluded.health_checked_at
                """,
                (
                    record.server_id,
                    record.user_id,
                    record.name,
                    record.transport,
                    int(record.enabled),
                    record.command,
                    json.dumps(record.args),
                    json.dumps(record.env_refs),
                    record.url,
                    record.auth_type,
                    record.auth_header_name,
                    record.auth_env_var,
                    int(record.allow_insecure_http),
                    int(record.allow_private_network),
                    json.dumps(record.allowed_roots),
                    record.status,
                    record.last_connected_at,
                    record.last_error,
                    created,
                    now,
                    record.registry_source,
                    record.registry_id,
                    record.listing_url,
                    record.description,
                    record.category,
                    record.icon_url,
                    record.package_type,
                    record.install_hint,
                    json.dumps(record.discovered_tools, default=str),
                    record.last_health_status,
                    record.health_checked_at,
                ),
            )
            self.db.connection.commit()
            return self.get(record.server_id, record.user_id)  # type: ignore[return-value]

    def get(self, server_id: str, user_id: str) -> Optional[MCPServerRecord]:
        with self.db.lock:
            row = self.db.connection.execute(
                "SELECT * FROM mcp_servers WHERE server_id = ? AND user_id = ?",
                (server_id, user_id),
            ).fetchone()
            return self._row_to_record(row) if row is not None else None

    def list_for_user(self, user_id: str) -> List[MCPServerRecord]:
        with self.db.lock:
            rows = self.db.connection.execute(
                "SELECT * FROM mcp_servers WHERE user_id = ? ORDER BY name ASC",
                (user_id,),
            ).fetchall()
            return [self._row_to_record(r) for r in rows]

    def update_status(
        self,
        server_id: str,
        user_id: str,
        status: str,
        last_error: str = "",
        last_connected_at: Optional[float] = None,
    ) -> None:
        with self.db.lock:
            if last_connected_at is not None:
                self.db.connection.execute(
                    """
                    UPDATE mcp_servers
                    SET status = ?, last_error = ?, last_connected_at = ?, updated_at = ?
                    WHERE server_id = ? AND user_id = ?
                    """,
                    (status, last_error, last_connected_at, time.time(), server_id, user_id),
                )
            else:
                self.db.connection.execute(
                    """
                    UPDATE mcp_servers
                    SET status = ?, last_error = ?, updated_at = ?
                    WHERE server_id = ? AND user_id = ?
                    """,
                    (status, last_error, time.time(), server_id, user_id),
                )
            self.db.connection.commit()

    def update_health(
        self,
        server_id: str,
        user_id: str,
        health_status: str,
        checked_at: Optional[float] = None,
    ) -> None:
        ts = checked_at if checked_at is not None else time.time()
        with self.db.lock:
            self.db.connection.execute(
                """
                UPDATE mcp_servers
                SET last_health_status = ?, health_checked_at = ?, updated_at = ?
                WHERE server_id = ? AND user_id = ?
                """,
                (health_status, ts, ts, server_id, user_id),
            )
            self.db.connection.commit()

    def update_discovered_tools(
        self,
        server_id: str,
        user_id: str,
        tools_meta: List[Dict[str, Any]],
    ) -> None:
        with self.db.lock:
            self.db.connection.execute(
                """
                UPDATE mcp_servers
                SET discovered_tools_json = ?, updated_at = ?
                WHERE server_id = ? AND user_id = ?
                """,
                (json.dumps(tools_meta, default=str), time.time(), server_id, user_id),
            )
            self.db.connection.commit()

    def delete(self, server_id: str, user_id: str) -> bool:
        with self.db.lock:
            cur = self.db.connection.execute(
                "DELETE FROM mcp_servers WHERE server_id = ? AND user_id = ?",
                (server_id, user_id),
            )
            self.db.connection.execute(
                "DELETE FROM mcp_tool_permissions WHERE server_id = ? AND user_id = ?",
                (server_id, user_id),
            )
            self.db.connection.execute(
                "DELETE FROM mcp_credentials WHERE server_id = ? AND user_id = ?",
                (server_id, user_id),
            )
            self.db.connection.commit()
            return cur.rowcount > 0


class PermissionRepository:
    """Persists user-specific tool permission and enablement policies."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def set_policy(self, record: ToolPermissionRecord) -> None:
        with self.db.lock:
            self.db.connection.execute(
                """
                INSERT INTO mcp_tool_permissions (
                    user_id, server_id, tool_name, category, auto_approve, enabled, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, server_id, tool_name) DO UPDATE SET
                    category=excluded.category,
                    auto_approve=excluded.auto_approve,
                    enabled=excluded.enabled,
                    updated_at=excluded.updated_at
                """,
                (
                    record.user_id,
                    record.server_id,
                    record.tool_name,
                    record.category,
                    int(record.auto_approve),
                    int(record.enabled),
                    record.updated_at or time.time(),
                ),
            )
            self.db.connection.commit()

    def set_tool_enabled(
        self,
        user_id: str,
        server_id: str,
        tool_name: str,
        enabled: bool,
        default_category: str = "MODIFICATION",
        default_auto_approve: bool = False,
    ) -> ToolPermissionRecord:
        with self.db.lock:
            existing = self.get_policy(user_id, server_id, tool_name)
            rec = ToolPermissionRecord(
                user_id=user_id,
                server_id=server_id,
                tool_name=tool_name,
                category=existing.category if existing is not None else default_category,
                auto_approve=existing.auto_approve if existing is not None else default_auto_approve,
                enabled=enabled,
                updated_at=time.time(),
            )
            self.set_policy(rec)
            return rec

    def get_policy(
        self,
        user_id: str,
        server_id: str,
        tool_name: str,
    ) -> Optional[ToolPermissionRecord]:
        with self.db.lock:
            row = self.db.connection.execute(
                """
                SELECT * FROM mcp_tool_permissions
                WHERE user_id = ? AND server_id = ? AND tool_name = ?
                """,
                (user_id, server_id, tool_name),
            ).fetchone()
            if row is None:
                return None
            return ToolPermissionRecord(
                user_id=row["user_id"],
                server_id=row["server_id"],
                tool_name=row["tool_name"],
                category=row["category"],
                auto_approve=bool(row["auto_approve"]),
                enabled=bool(row["enabled"]) if "enabled" in row.keys() else True,
                updated_at=row["updated_at"],
            )

    def list_for_user(self, user_id: str) -> List[ToolPermissionRecord]:
        with self.db.lock:
            rows = self.db.connection.execute(
                "SELECT * FROM mcp_tool_permissions WHERE user_id = ? ORDER BY server_id, tool_name",
                (user_id,),
            ).fetchall()
            return [
                ToolPermissionRecord(
                    user_id=r["user_id"],
                    server_id=r["server_id"],
                    tool_name=r["tool_name"],
                    category=r["category"],
                    auto_approve=bool(r["auto_approve"]),
                    enabled=bool(r["enabled"]) if "enabled" in r.keys() else True,
                    updated_at=r["updated_at"],
                )
                for r in rows
            ]


class ApprovalRepository:
    """Persists human-in-the-loop tool approvals strictly bound to user/thread/tool/args hash."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    @staticmethod
    def _row_to_record(row: Any) -> ApprovalRecord:
        return ApprovalRecord(
            approval_id=row["approval_id"],
            user_id=row["user_id"],
            thread_id=row["thread_id"],
            server_id=row["server_id"],
            tool_name=row["tool_name"],
            arguments_hash=row["arguments_hash"],
            masked_arguments=row["masked_arguments"],
            category=row["category"],
            reason=row["reason"],
            status=row["status"],
            created_at=row["created_at"],
            decided_at=row["decided_at"],
        )

    def create(self, record: ApprovalRecord) -> ApprovalRecord:
        with self.db.lock:
            self.db.connection.execute(
                """
                INSERT INTO mcp_approvals (
                    approval_id, user_id, thread_id, server_id, tool_name,
                    arguments_hash, masked_arguments, category, reason,
                    status, created_at, decided_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.approval_id,
                    record.user_id,
                    record.thread_id,
                    record.server_id,
                    record.tool_name,
                    record.arguments_hash,
                    record.masked_arguments,
                    record.category,
                    record.reason,
                    record.status,
                    record.created_at or time.time(),
                    record.decided_at,
                ),
            )
            self.db.connection.commit()
            return record

    def get(self, approval_id: str, user_id: str) -> Optional[ApprovalRecord]:
        with self.db.lock:
            row = self.db.connection.execute(
                "SELECT * FROM mcp_approvals WHERE approval_id = ? AND user_id = ?",
                (approval_id, user_id),
            ).fetchone()
            return self._row_to_record(row) if row is not None else None

    def list_pending(
        self,
        user_id: str,
        thread_id: Optional[str] = None,
    ) -> List[ApprovalRecord]:
        with self.db.lock:
            if thread_id:
                rows = self.db.connection.execute(
                    """
                    SELECT * FROM mcp_approvals
                    WHERE user_id = ? AND thread_id = ? AND status = 'PENDING'
                    ORDER BY created_at DESC
                    """,
                    (user_id, thread_id),
                ).fetchall()
            else:
                rows = self.db.connection.execute(
                    """
                    SELECT * FROM mcp_approvals
                    WHERE user_id = ? AND status = 'PENDING'
                    ORDER BY created_at DESC
                    """,
                    (user_id,),
                ).fetchall()
            return [self._row_to_record(r) for r in rows]

    def decide(
        self,
        approval_id: str,
        user_id: str,
        status: str,
    ) -> Optional[ApprovalRecord]:
        with self.db.lock:
            cur = self.db.connection.execute(
                """
                UPDATE mcp_approvals
                SET status = ?, decided_at = ?
                WHERE approval_id = ? AND user_id = ?
                """,
                (status, time.time(), approval_id, user_id),
            )
            self.db.connection.commit()
            if cur.rowcount == 0:
                return None
            return self.get(approval_id, user_id)


class AuditLogRepository:
    """Stores user-scoped MCP tool execution audit records."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def record(self, entry: AuditLogRecord) -> None:
        with self.db.lock:
            self.db.connection.execute(
                """
                INSERT INTO mcp_audit_logs (
                    execution_id, user_id, thread_id, server_id, server_name,
                    tool_name, status, masked_arguments, result_summary,
                    error, started_at, completed_at, duration_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(execution_id) DO UPDATE SET
                    status=excluded.status,
                    result_summary=excluded.result_summary,
                    error=excluded.error,
                    completed_at=excluded.completed_at,
                    duration_ms=excluded.duration_ms
                """,
                (
                    entry.execution_id,
                    entry.user_id,
                    entry.thread_id,
                    entry.server_id,
                    entry.server_name,
                    entry.tool_name,
                    entry.status,
                    entry.masked_arguments,
                    entry.result_summary,
                    entry.error,
                    entry.started_at,
                    entry.completed_at,
                    entry.duration_ms,
                ),
            )
            self.db.connection.commit()

    def list_recent(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        with self.db.lock:
            rows = self.db.connection.execute(
                """
                SELECT * FROM mcp_audit_logs
                WHERE user_id = ?
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]


class CredentialRepository:
    """Stores Fernet-encrypted MCP credentials strictly partitioned by (user_id, server_id)."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    @staticmethod
    def _row_to_record(row: Any) -> EncryptedCredentialRecord:
        return EncryptedCredentialRecord(
            credential_id=row["credential_id"],
            user_id=row["user_id"],
            server_id=row["server_id"],
            auth_type=row["auth_type"],
            header_name=row["header_name"],
            encrypted_payload=row["encrypted_payload"],
            key_fingerprint=row["key_fingerprint"],
            expires_at=row["expires_at"],
            scopes=row["scopes"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def upsert(self, record: EncryptedCredentialRecord) -> EncryptedCredentialRecord:
        now = time.time()
        created = record.created_at or now
        with self.db.lock:
            self.db.connection.execute(
                """
                INSERT INTO mcp_credentials (
                    credential_id, user_id, server_id, auth_type, header_name,
                    encrypted_payload, key_fingerprint, expires_at, scopes,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, server_id, auth_type) DO UPDATE SET
                    header_name=excluded.header_name,
                    encrypted_payload=excluded.encrypted_payload,
                    key_fingerprint=excluded.key_fingerprint,
                    expires_at=excluded.expires_at,
                    scopes=excluded.scopes,
                    updated_at=excluded.updated_at
                """,
                (
                    record.credential_id,
                    record.user_id,
                    record.server_id,
                    record.auth_type,
                    record.header_name,
                    record.encrypted_payload,
                    record.key_fingerprint,
                    record.expires_at,
                    record.scopes,
                    created,
                    now,
                ),
            )
            self.db.connection.commit()
            saved = self.get(record.user_id, record.server_id, record.auth_type)
            return saved or record

    def get(
        self,
        user_id: str,
        server_id: str,
        auth_type: str,
    ) -> Optional[EncryptedCredentialRecord]:
        with self.db.lock:
            row = self.db.connection.execute(
                """
                SELECT * FROM mcp_credentials
                WHERE user_id = ? AND server_id = ? AND auth_type = ?
                """,
                (user_id, server_id, auth_type),
            ).fetchone()
            return self._row_to_record(row) if row is not None else None

    def list_for_server(self, user_id: str, server_id: str) -> List[EncryptedCredentialRecord]:
        with self.db.lock:
            rows = self.db.connection.execute(
                "SELECT * FROM mcp_credentials WHERE user_id = ? AND server_id = ?",
                (user_id, server_id),
            ).fetchall()
            return [self._row_to_record(r) for r in rows]

    def list_for_user(self, user_id: str) -> List[EncryptedCredentialRecord]:
        with self.db.lock:
            rows = self.db.connection.execute(
                "SELECT * FROM mcp_credentials WHERE user_id = ?",
                (user_id,),
            ).fetchall()
            return [self._row_to_record(r) for r in rows]

    def delete(
        self,
        user_id: str,
        server_id: str,
        auth_type: Optional[str] = None,
    ) -> int:
        with self.db.lock:
            if auth_type:
                cur = self.db.connection.execute(
                    "DELETE FROM mcp_credentials WHERE user_id = ? AND server_id = ? AND auth_type = ?",
                    (user_id, server_id, auth_type),
                )
            else:
                cur = self.db.connection.execute(
                    "DELETE FROM mcp_credentials WHERE user_id = ? AND server_id = ?",
                    (user_id, server_id),
                )
            self.db.connection.commit()
            return cur.rowcount


class RegistryCacheRepository:
    """SQLite TTL cache for normalized MCP registry responses."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def get(self, cache_key: str) -> Optional[tuple[str, float]]:
        now = time.time()
        with self.db.lock:
            row = self.db.connection.execute(
                "SELECT payload_json, fetched_at, ttl_sec FROM mcp_registry_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if row is None:
                return None
            fetched_at = float(row["fetched_at"])
            ttl_sec = int(row["ttl_sec"])
            if now - fetched_at > ttl_sec:
                return None
            return str(row["payload_json"]), fetched_at

    def put(self, cache_key: str, source: str, payload_json: str, ttl_sec: int = 300) -> None:
        with self.db.lock:
            self.db.connection.execute(
                """
                INSERT INTO mcp_registry_cache (cache_key, source, payload_json, fetched_at, ttl_sec)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    source=excluded.source,
                    payload_json=excluded.payload_json,
                    fetched_at=excluded.fetched_at,
                    ttl_sec=excluded.ttl_sec
                """,
                (cache_key, source, payload_json, time.time(), max(1, int(ttl_sec))),
            )
            self.db.connection.commit()

    def clear(self, source: Optional[str] = None) -> None:
        with self.db.lock:
            if source:
                self.db.connection.execute(
                    "DELETE FROM mcp_registry_cache WHERE source = ?",
                    (source,),
                )
            else:
                self.db.connection.execute("DELETE FROM mcp_registry_cache")
            self.db.connection.commit()
