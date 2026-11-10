"""Non-destructive legacy workspace migration and account claiming helpers."""

from datetime import datetime, timezone
import json
from typing import Dict

from sqlalchemy import inspect, select, text

from qna_bot.core.logging_config import get_logger
from qna_bot.database.session import SQLAlchemyDatabase
from qna_bot.models.conversation import Conversation, Message
from qna_bot.models.user import User

logger = get_logger(__name__)

LEGACY_USER_ID = "local_user"


def _legacy_table_has_data(db: SQLAlchemyDatabase) -> bool:
    with db.engine.connect() as conn:
        inspector = inspect(conn)
        tables = set(inspector.get_table_names())
        for tbl in ("threads", "thread_messages", "mcp_servers", "mcp_credentials", "mcp_audit_logs"):
            if tbl in tables:
                cnt = conn.execute(text(f"SELECT COUNT(*) FROM {tbl} WHERE user_id = :u"), {"u": LEGACY_USER_ID}).scalar()
                if cnt and int(cnt) > 0:
                    return True
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
        tables = set(inspect(conn).get_table_names())
        if "threads" in tables:
            legacy_threads = conn.execute(
                text(
                    "SELECT thread_id, user_id, title, chat_mode, is_pinned, created_at, updated_at "
                    "FROM threads"
                )
            ).mappings().all()
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
                    created_dt = datetime.fromtimestamp(float(row["created_at"] or 0), tz=timezone.utc)
                    updated_dt = datetime.fromtimestamp(float(row["updated_at"] or 0), tz=timezone.utc)
                    session.add(
                        Conversation(
                            id=tid,
                            user_id=uid,
                            title=str(row["title"] or "New Conversation")[:255],
                            agent_mode=str(row["chat_mode"] or "Auto (Agentic)")[:64],
                            is_pinned=bool(row["is_pinned"]),
                            created_at=created_dt,
                            updated_at=updated_dt,
                        )
                    )
                    summary["threads_migrated"] += 1
            session.flush()

        if "thread_messages" in tables:
            legacy_msgs = conn.execute(
                text(
                    "SELECT id, thread_id, user_id, role, content, sources_json, "
                    "web_sources_json, tool_events_json, timestamp FROM thread_messages ORDER BY id ASC"
                )
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
                    "legacy_thread_msg_id": int(row["id"]),
                    "sources": json.loads(row["sources_json"] or "[]"),
                    "web_sources": json.loads(row["web_sources_json"] or "[]"),
                    "tool_events": json.loads(row["tool_events_json"] or "[]"),
                }
                ts = datetime.fromtimestamp(float(row["timestamp"] or 0), tz=timezone.utc)
                session.add(
                    Message(
                        conversation_id=tid,
                        user_id=uid,
                        role=str(row["role"] or "user"),
                        content=str(row["content"] or ""),
                        metadata_json=json.dumps(meta),
                        created_at=ts,
                    )
                )
                summary["messages_migrated"] += 1

    return summary


def has_unclaimed_legacy_data(db: SQLAlchemyDatabase) -> bool:
    """Return True if legacy local_user data exists and can be claimed by a real account."""
    with db.engine.connect() as conn:
        inspector = inspect(conn)
        tables = set(inspector.get_table_names())
        for tbl in ("conversations", "threads", "mcp_servers", "uploaded_documents"):
            if tbl in tables:
                cnt = conn.execute(
                    text(f"SELECT COUNT(*) FROM {tbl} WHERE user_id = :u"),
                    {"u": LEGACY_USER_ID},
                ).scalar()
                if cnt and int(cnt) > 0:
                    return True
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
        tables = set(inspect(conn).get_table_names())
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
            if tbl in tables:
                res = conn.execute(
                    text(f"UPDATE {tbl} SET user_id = :target WHERE user_id = :legacy"),
                    {"target": target_user_id, "legacy": LEGACY_USER_ID},
                )
                moved[tbl] = int(res.rowcount or 0)

        # Keep LangGraph checkpoint thread_id prefix aligned for legacy user threads
        for ck_tbl in ("checkpoints", "writes"):
            if ck_tbl in tables:
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
