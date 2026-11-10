"""SQLite database connection and schema initialization."""

from pathlib import Path
import sqlite3
import threading
from typing import Optional

from qna_bot.config.settings import get_settings
from qna_bot.core.exceptions import MemoryPersistenceError
from qna_bot.core.logging_config import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """Thread-safe SQLite connection manager shared across threads, MCP registry, approvals, and audit logs."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        settings = get_settings()
        self.db_path = Path(db_path or settings.sqlite_db_path)
        self.lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._initialize()

    def _initialize(self) -> None:
        try:
            if str(self.db_path) != ":memory:":
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON;")
            self._conn.execute("PRAGMA busy_timeout = 10000;")
            if str(self.db_path) != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL;")
                self._conn.execute("PRAGMA synchronous = NORMAL;")
            self._create_tables()
        except Exception as exc:
            logger.exception("Failed initializing SQLite database at %s", self.db_path)
            raise MemoryPersistenceError(f"Database initialization failed: {exc}") from exc

    def _create_tables(self) -> None:
        assert self._conn is not None
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS threads (
                thread_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                chat_mode TEXT NOT NULL,
                is_pinned INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        # Safe additive migration for pre-existing threads tables
        cols = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(threads)").fetchall()
        }
        if "is_pinned" not in cols:
            self._conn.execute(
                "ALTER TABLE threads ADD COLUMN is_pinned INTEGER NOT NULL DEFAULT 0"
            )

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS thread_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources_json TEXT NOT NULL DEFAULT '[]',
                web_sources_json TEXT NOT NULL DEFAULT '[]',
                tool_events_json TEXT NOT NULL DEFAULT '[]',
                image_b64 TEXT,
                image_mime TEXT,
                timestamp REAL NOT NULL,
                FOREIGN KEY(thread_id) REFERENCES threads(thread_id)
            )
            """
        )
        msg_cols = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(thread_messages)").fetchall()
        }
        if "tool_events_json" not in msg_cols:
            self._conn.execute(
                "ALTER TABLE thread_messages ADD COLUMN tool_events_json TEXT NOT NULL DEFAULT '[]'"
            )

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mcp_servers (
                server_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                transport TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                command TEXT NOT NULL DEFAULT '',
                args_json TEXT NOT NULL DEFAULT '[]',
                env_refs_json TEXT NOT NULL DEFAULT '{}',
                url TEXT NOT NULL DEFAULT '',
                auth_type TEXT NOT NULL DEFAULT 'none',
                auth_header_name TEXT NOT NULL DEFAULT 'Authorization',
                auth_env_var TEXT NOT NULL DEFAULT '',
                allow_insecure_http INTEGER NOT NULL DEFAULT 0,
                allow_private_network INTEGER NOT NULL DEFAULT 0,
                allowed_roots_json TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL DEFAULT 'disconnected',
                last_connected_at REAL,
                last_error TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                registry_source TEXT NOT NULL DEFAULT 'manual',
                registry_id TEXT NOT NULL DEFAULT '',
                listing_url TEXT NOT NULL DEFAULT '',
                description TEXT NOT NULL DEFAULT '',
                category TEXT NOT NULL DEFAULT 'General',
                icon_url TEXT NOT NULL DEFAULT '',
                package_type TEXT NOT NULL DEFAULT '',
                install_hint TEXT NOT NULL DEFAULT '',
                discovered_tools_json TEXT NOT NULL DEFAULT '[]',
                last_health_status TEXT NOT NULL DEFAULT 'unknown',
                health_checked_at REAL
            )
            """
        )
        # Safe additive migrations for existing mcp_servers databases
        srv_cols = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(mcp_servers)").fetchall()
        }
        for col_name, col_ddl in (
            ("registry_source", "TEXT NOT NULL DEFAULT 'manual'"),
            ("registry_id", "TEXT NOT NULL DEFAULT ''"),
            ("listing_url", "TEXT NOT NULL DEFAULT ''"),
            ("description", "TEXT NOT NULL DEFAULT ''"),
            ("category", "TEXT NOT NULL DEFAULT 'General'"),
            ("icon_url", "TEXT NOT NULL DEFAULT ''"),
            ("package_type", "TEXT NOT NULL DEFAULT ''"),
            ("install_hint", "TEXT NOT NULL DEFAULT ''"),
            ("discovered_tools_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("last_health_status", "TEXT NOT NULL DEFAULT 'unknown'"),
            ("health_checked_at", "REAL"),
        ):
            if col_name not in srv_cols:
                self._conn.execute(f"ALTER TABLE mcp_servers ADD COLUMN {col_name} {col_ddl}")

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mcp_tool_permissions (
                user_id TEXT NOT NULL,
                server_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                category TEXT NOT NULL,
                auto_approve INTEGER NOT NULL DEFAULT 0,
                enabled INTEGER NOT NULL DEFAULT 1,
                updated_at REAL NOT NULL,
                PRIMARY KEY (user_id, server_id, tool_name)
            )
            """
        )
        perm_cols = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(mcp_tool_permissions)").fetchall()
        }
        if "enabled" not in perm_cols:
            self._conn.execute(
                "ALTER TABLE mcp_tool_permissions ADD COLUMN enabled INTEGER NOT NULL DEFAULT 1"
            )

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mcp_credentials (
                credential_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                server_id TEXT NOT NULL,
                auth_type TEXT NOT NULL,
                header_name TEXT NOT NULL DEFAULT 'Authorization',
                encrypted_payload TEXT NOT NULL,
                key_fingerprint TEXT NOT NULL DEFAULT '',
                expires_at REAL,
                scopes TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE (user_id, server_id, auth_type)
            )
            """
        )

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mcp_registry_cache (
                cache_key TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                fetched_at REAL NOT NULL,
                ttl_sec INTEGER NOT NULL DEFAULT 300
            )
            """
        )

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mcp_approvals (
                approval_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                server_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                arguments_hash TEXT NOT NULL,
                masked_arguments TEXT NOT NULL,
                category TEXT NOT NULL,
                reason TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at REAL NOT NULL,
                decided_at REAL
            )
            """
        )

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mcp_audit_logs (
                execution_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                server_id TEXT NOT NULL,
                server_name TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                status TEXT NOT NULL,
                masked_arguments TEXT NOT NULL,
                result_summary TEXT NOT NULL,
                error TEXT NOT NULL DEFAULT '',
                started_at REAL NOT NULL,
                completed_at REAL,
                duration_ms REAL
            )
            """
        )

        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_threads_user ON threads(user_id, is_pinned DESC, updated_at DESC)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mcp_servers_user ON mcp_servers(user_id, name)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mcp_approvals_user ON mcp_approvals(user_id, thread_id, status)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mcp_audit_user ON mcp_audit_logs(user_id, started_at DESC)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mcp_creds_user_srv ON mcp_credentials(user_id, server_id)"
        )
        self._conn.commit()

    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise MemoryPersistenceError("Database connection is closed.")
        return self._conn

    def close(self) -> None:
        with self.lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
