"""Non-destructive legacy workspace migration and account claiming helpers."""

from datetime import datetime, timezone
import json
import time
from typing import Any, Dict, List

from sqlalchemy import inspect, select, text
from sqlalchemy.exc import SQLAlchemyError

from qna_bot.core.logging_config import get_logger
from qna_bot.database.session import SQLAlchemyDatabase
from qna_bot.models.conversation import Conversation, Message
from qna_bot.models.user import User

logger = get_logger(__name__)

LEGACY_USER_ID = "local_user"


def _table_columns(inspector: Any, table_name: str) -> set[str]:
    try:
        return {str(c["name"]) for c in inspector.get_columns(table_name)}
    except Exception:
        return set()


def _safe_json_list(raw: Any) -> List[Any]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def _legacy_table_has_data(db: SQLAlchemyDatabase) -> bool:
    try:
        with db.engine.connect() as conn:
            inspector = inspect(conn)
            tables = set(inspector.get_table_names())
            for tbl in ("threads", "thread_messages", "mcp_servers", "mcp_credentials", "mcp_audit_logs"):
                if tbl in tables and "user_id" in _table_columns(inspector, tbl):
                    cnt = conn.execute(
                        text(f"SELECT COUNT(*) FROM {tbl} WHERE user_id = :u"),
                        {"u": LEGACY_USER_ID},
                    ).scalar()
                    if cnt and int(cnt) > 0:
                        return True
    except SQLAlchemyError as exc:
        logger.warning("Skipping legacy workspace probe due to SQLite error: %s", exc)
    return False


def ensure_legacy_workspace_synced(db: SQLAlchemyDatabase) -> Dict[str, int]:
    """Ensure legacy local_user SQLite threads/messages exist cleanly in normalized ORM tables."""
    summary = {"threads_migrated": 0, "messages_migrated": 0, "placeholder_created": 0}
    if not _legacy_table_has_data(db):
        return summary

    with db.session_scope() as session:
        legacy_user = session.get(User, LEGACY_USER_ID)
        if legacy_user is None:
            session.add(
                User(
                    id=LEGACY_USER_ID,
                    email="local_user@local.qnagpt",
                    password_hash=None,  # Locked: cannot log in via password; claimed explicitly by real account
                    full_name="Legacy Local Workspace",
                    is_active=True,
                    is_verified=False,
                )
            )
            session.flush()
            summary["placeholder_created"] = 1

        conn = session.connection()
        inspector = inspect(conn)
        tables = set(inspector.get_table_names())
        now_ts = time.time()

        if "threads" in tables:
            tcols = _table_columns(inspector, "threads")
            desired_tcols = [
                c
                for c in ("thread_id", "user_id", "title", "chat_mode", "is_pinned", "created_at", "updated_at")
                if c in tcols
            ]
            if "thread_id" in desired_tcols and "user_id" in desired_tcols:
                col_sql = ", ".join(desired_tcols)
                legacy_threads = conn.execute(text(f"SELECT {col_sql} FROM threads")).mappings().all()
                for row in legacy_threads:
                    uid = str(row["user_id"])
                    if session.get(User, uid) is None:
                        session.add(
                            User(
                                id=uid,
                                email=f"{uid}@local.qnagpt",
                                password_hash=None,
                                full_name=f"Workspace {uid}",
                                is_active=True,
                                is_verified=False,
                            )
                        )
                        session.flush()

                    tid = str(row["thread_id"])
                    if session.get(Conversation, tid) is None:
                        created_raw = float(row.get("created_at") or now_ts)
                        updated_raw = float(row.get("updated_at") or created_raw)
                        session.add(
                            Conversation(
                                id=tid,
                                user_id=uid,
                                title=str(row.get("title") or "New Conversation")[:255],
                                agent_mode=str(row.get("chat_mode") or "Auto (Agentic)")[:64],
                                is_pinned=bool(row.get("is_pinned", False)),
                                created_at=datetime.fromtimestamp(created_raw, tz=timezone.utc),
                                updated_at=datetime.fromtimestamp(updated_raw, tz=timezone.utc),
                            )
                        )
                        summary["threads_migrated"] += 1
                session.flush()

        if "thread_messages" in tables:
            mcols = _table_columns(inspector, "thread_messages")
            desired_mcols = [
                c
                for c in (
                    "id",
                    "thread_id",
                    "user_id",
                    "role",
                    "content",
                    "sources_json",
                    "web_sources_json",
                    "tool_events_json",
                    "timestamp",
                )
                if c in mcols
            ]
            if {"id", "thread_id", "user_id"}.issubset(desired_mcols):
                mcol_sql = ", ".join(desired_mcols)
                legacy_msgs = conn.execute(
                    text(f"SELECT {mcol_sql} FROM thread_messages ORDER BY id ASC")
                ).mappings().all()
                existing_legacy_ids: set[int] = set()
                for raw_meta in session.scalars(select(Message.metadata_json)).all():
                    if not raw_meta:
                        continue
                    try:
                        parsed_meta = json.loads(raw_meta)
                        if isinstance(parsed_meta, dict) and parsed_meta.get("legacy_thread_msg_id") is not None:
                            existing_legacy_ids.add(int(parsed_meta["legacy_thread_msg_id"]))
                    except (ValueError, TypeError):
                        continue

                for row in legacy_msgs:
                    row_id = int(row["id"])
                    if row_id in existing_legacy_ids:
                        continue
                    tid = str(row["thread_id"])
                    uid = str(row["user_id"])
                    if session.get(Conversation, tid) is None or session.get(User, uid) is None:
                        continue
                    meta = {
                        "legacy_thread_msg_id": row_id,
                        "sources": _safe_json_list(row.get("sources_json")),
                        "web_sources": _safe_json_list(row.get("web_sources_json")),
                        "tool_events": _safe_json_list(row.get("tool_events_json")),
                    }
                    ts_val = float(row.get("timestamp") or now_ts)
                    session.add(
                        Message(
                            conversation_id=tid,
                            user_id=uid,
                            role=str(row.get("role") or "user"),
                            content=str(row.get("content") or ""),
                            metadata_json=json.dumps(meta),
                            created_at=datetime.fromtimestamp(ts_val, tz=timezone.utc),
                        )
                    )
                    summary["messages_migrated"] += 1

    return summary


def has_unclaimed_legacy_data(db: SQLAlchemyDatabase) -> bool:
    """Return True if legacy local_user data exists and can be claimed by a real account."""
    try:
        with db.engine.connect() as conn:
            inspector = inspect(conn)
            tables = set(inspector.get_table_names())
            for tbl in ("conversations", "threads", "mcp_servers", "uploaded_documents"):
                if tbl in tables and "user_id" in _table_columns(inspector, tbl):
                    cnt = conn.execute(
                        text(f"SELECT COUNT(*) FROM {tbl} WHERE user_id = :u"),
                        {"u": LEGACY_USER_ID},
                    ).scalar()
                    if cnt and int(cnt) > 0:
                        return True
    except SQLAlchemyError:
        return False
    return False


def claim_legacy_workspace(db: SQLAlchemyDatabase, target_user_id: str) -> Dict[str, int]:
    """Reassign all legacy local_user records atomically to a real authenticated user account."""
    if not target_user_id or target_user_id == LEGACY_USER_ID:
        raise ValueError("Target user must be a distinct authenticated user account.")

    moved: Dict[str, int] = {}
    with db.session_scope() as session:
        target = session.get(User, target_user_id)
        if target is None:
            raise ValueError(f"Authenticated user '{target_user_id}' does not exist.")

        conn = session.connection()
        inspector = inspect(conn)
        tables = set(inspector.get_table_names())
        user_owned_tables = (
            "conversations",
            "messages",
            "uploaded_documents",
            "agent_executions",
            "mcp_execution_logs",
            "mcp_servers",
            "mcp_tool_permissions",
            "mcp_credentials",
            "mcp_approvals",
            "mcp_audit_logs",
            "audit_logs",
            "threads",
            "thread_messages",
        )
        for tbl in user_owned_tables:
            if tbl in tables and "user_id" in _table_columns(inspector, tbl):
                res = conn.execute(
                    text(f"UPDATE {tbl} SET user_id = :target WHERE user_id = :legacy"),
                    {"target": target_user_id, "legacy": LEGACY_USER_ID},
                )
                moved[tbl] = int(res.rowcount or 0)

        for ck_tbl in ("checkpoints", "writes"):
            if ck_tbl in tables and "thread_id" in _table_columns(inspector, ck_tbl):
                conn.execute(
                    text(
                        f"UPDATE {ck_tbl} "
                        f"SET thread_id = :new_prefix || substr(thread_id, :prefix_len) "
                        f"WHERE thread_id LIKE :like_pat"
                    ),
                    {
                        "new_prefix": f"{target_user_id}:",
                        "prefix_len": len(f"{LEGACY_USER_ID}:") + 1,
                        "like_pat": f"{LEGACY_USER_ID}:%",
                    },
                )

    return moved
