"""SQLite conversation persistence, thread ownership enforcement, and LangGraph SqliteSaver."""

from datetime import datetime
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional
import uuid

from langgraph.checkpoint.sqlite import SqliteSaver

from qna_bot.config.settings import get_settings
from qna_bot.core.exceptions import MemoryPersistenceError
from qna_bot.core.logging_config import get_logger

logger = get_logger(__name__)

_TITLE_MAX_LEN = 120


def classify_time_bucket(timestamp: float, now_ts: Optional[float] = None) -> str:
    """Group conversation timestamps into UI history sections."""
    now_dt = datetime.fromtimestamp(now_ts if now_ts is not None else time.time()).date()
    msg_dt = datetime.fromtimestamp(timestamp).date()
    delta_days = (now_dt - msg_dt).days
    if delta_days <= 0:
        return "Today"
    if delta_days == 1:
        return "Yesterday"
    if delta_days <= 7:
        return "Previous 7 Days"
    if delta_days <= 30:
        return "Previous 30 Days"
    return "Older"


class ConversationMemoryManager:
    """Manages SQLite checkpointing and user-scoped conversation threads."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        settings = get_settings()
        self.db_path = Path(db_path or settings.sqlite_db_path)
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._checkpointer: Optional[SqliteSaver] = None
        self._initialize_database()

    def _initialize_database(self) -> None:
        try:
            if str(self.db_path) != ":memory:":
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._checkpointer = SqliteSaver(self._conn)
            self._checkpointer.setup()

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
                "CREATE INDEX IF NOT EXISTS idx_threads_user ON threads(user_id, is_pinned DESC, updated_at DESC)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_msgs_thread ON thread_messages(thread_id, user_id, id)"
            )
            self._conn.commit()
        except Exception as exc:
            logger.exception("Failed to initialize SQLite memory at %s", self.db_path)
            raise MemoryPersistenceError(f"SQLite initialization failed: {exc}") from exc

    @property
    def checkpointer(self) -> SqliteSaver:
        if self._checkpointer is None:
            raise MemoryPersistenceError("SQLite checkpointer is not initialized.")
        return self._checkpointer

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
                self._checkpointer = None

    @staticmethod
    def _validate_ids(thread_id: str, user_id: str) -> tuple[str, str]:
        tid = (thread_id or "").strip()
        uid = (user_id or "").strip()
        if not tid:
            raise MemoryPersistenceError("Invalid empty thread_id.")
        if not uid:
            raise MemoryPersistenceError("Invalid empty user_id.")
        return tid, uid

    def _verify_thread_ownership(self, thread_id: str, user_id: str) -> None:
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT user_id FROM threads WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        if row is not None and row["user_id"] != user_id:
            raise MemoryPersistenceError(
                "Unauthorized access: thread belongs to another user."
            )

    def create_new_thread(
        self,
        user_id: str,
        title: str = "New Conversation",
        chat_mode: str = "Auto (Agentic)",
        thread_id: Optional[str] = None,
    ) -> str:
        """Create and persist a new conversation thread owned by user_id."""
        uid = (user_id or "").strip()
        if not uid:
            raise MemoryPersistenceError("A non-empty user_id is required to create a thread.")

        tid = (thread_id or f"conversation_{uuid.uuid4().hex[:12]}").strip()
        now = time.time()
        with self._lock:
            try:
                assert self._conn is not None
                self._verify_thread_ownership(tid, uid)
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO threads (
                        thread_id, user_id, title, chat_mode, is_pinned, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 0, ?, ?)
                    """,
                    (tid, uid, title, chat_mode, now, now),
                )
                self._conn.commit()
                return tid
            except MemoryPersistenceError:
                raise
            except Exception as exc:
                raise MemoryPersistenceError(f"Failed creating thread: {exc}") from exc

    def ensure_thread(
        self,
        thread_id: str,
        user_id: str,
        title: str = "New Conversation",
        chat_mode: str = "Auto (Agentic)",
    ) -> str:
        return self.create_new_thread(
            user_id=user_id,
            title=title,
            chat_mode=chat_mode,
            thread_id=thread_id,
        )

    def rename_thread(self, thread_id: str, user_id: str, new_title: str) -> bool:
        """Rename a conversation thread owned by user_id."""
        tid, uid = self._validate_ids(thread_id, user_id)
        cleaned = (new_title or "").strip()
        if not cleaned:
            raise MemoryPersistenceError("Thread title cannot be empty.")
        with self._lock:
            try:
                assert self._conn is not None
                self._verify_thread_ownership(tid, uid)
                cur = self._conn.execute(
                    """
                    UPDATE threads
                    SET title = ?, updated_at = ?
                    WHERE thread_id = ? AND user_id = ?
                    """,
                    (cleaned[:80], time.time(), tid, uid),
                )
                self._conn.commit()
                return cur.rowcount > 0
            except MemoryPersistenceError:
                raise
            except Exception as exc:
                raise MemoryPersistenceError(f"Failed renaming thread: {exc}") from exc

    def set_thread_pinned(self, thread_id: str, user_id: str, pinned: bool) -> bool:
        """Pin or unpin a conversation thread owned by user_id."""
        tid, uid = self._validate_ids(thread_id, user_id)
        with self._lock:
            try:
                assert self._conn is not None
                self._verify_thread_ownership(tid, uid)
                cur = self._conn.execute(
                    "UPDATE threads SET is_pinned = ? WHERE thread_id = ? AND user_id = ?",
                    (1 if pinned else 0, tid, uid),
                )
                self._conn.commit()
                return cur.rowcount > 0
            except MemoryPersistenceError:
                raise
            except Exception as exc:
                raise MemoryPersistenceError(f"Failed pinning thread: {exc}") from exc

    def update_thread_title(
        self,
        thread_id: str,
        user_id: str,
        first_prompt: str,
        chat_mode: Optional[str] = None,
    ) -> None:
        tid, uid = self._validate_ids(thread_id, user_id)
        cleaned = (first_prompt or "").strip()
        title = (
            f"{cleaned[:_TITLE_MAX_LEN]}..."
            if len(cleaned) > _TITLE_MAX_LEN
            else (cleaned or "Conversation")
        )
        now = time.time()
        with self._lock:
            try:
                assert self._conn is not None
                self._verify_thread_ownership(tid, uid)
                if chat_mode:
                    self._conn.execute(
                        """
                        UPDATE threads
                        SET title = CASE WHEN title = 'New Conversation' THEN ? ELSE title END,
                            chat_mode = ?,
                            updated_at = ?
                        WHERE thread_id = ? AND user_id = ?
                        """,
                        (title, chat_mode, now, tid, uid),
                    )
                else:
                    self._conn.execute(
                        """
                        UPDATE threads
                        SET title = CASE WHEN title = 'New Conversation' THEN ? ELSE title END,
                            updated_at = ?
                        WHERE thread_id = ? AND user_id = ?
                        """,
                        (title, now, tid, uid),
                    )
                self._conn.commit()
            except MemoryPersistenceError:
                raise
            except Exception as exc:
                raise MemoryPersistenceError(f"Failed updating thread metadata: {exc}") from exc

    def append_message(
        self,
        thread_id: str,
        user_id: str,
        message: Dict[str, Any],
    ) -> None:
        """Persist a single UI chat message (with citations/tool events/attachments) scoped to user_id."""
        tid, uid = self._validate_ids(thread_id, user_id)
        with self._lock:
            try:
                assert self._conn is not None
                self.ensure_thread(tid, uid)
                self._conn.execute(
                    """
                    INSERT INTO thread_messages (
                        thread_id, user_id, role, content,
                        sources_json, web_sources_json, tool_events_json,
                        image_b64, image_mime, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tid,
                        uid,
                        str(message.get("role", "user")),
                        str(message.get("content", "")),
                        json.dumps(message.get("sources") or []),
                        json.dumps(message.get("web_sources") or []),
                        json.dumps(message.get("tool_events") or []),
                        message.get("image_b64"),
                        message.get("image_mime"),
                        float(message.get("timestamp") or time.time()),
                    ),
                )
                self._conn.execute(
                    "UPDATE threads SET updated_at = ? WHERE thread_id = ? AND user_id = ?",
                    (time.time(), tid, uid),
                )
                self._conn.commit()
            except MemoryPersistenceError:
                raise
            except Exception as exc:
                raise MemoryPersistenceError(f"Failed persisting message: {exc}") from exc

    def replace_thread_messages(
        self,
        thread_id: str,
        user_id: str,
        messages: List[Dict[str, Any]],
    ) -> None:
        """Replace UI messages for a thread (e.g., on clear or regeneration)."""
        tid, uid = self._validate_ids(thread_id, user_id)
        with self._lock:
            try:
                assert self._conn is not None
                self._verify_thread_ownership(tid, uid)
                self._conn.execute(
                    "DELETE FROM thread_messages WHERE thread_id = ? AND user_id = ?",
                    (tid, uid),
                )
                for msg in messages:
                    self.append_message(tid, uid, msg)
                self._conn.commit()
            except MemoryPersistenceError:
                raise
            except Exception as exc:
                raise MemoryPersistenceError(f"Failed updating thread messages: {exc}") from exc

    def retrieve_all_threads(
        self,
        user_id: str,
        search_query: str = "",
    ) -> List[Dict[str, Any]]:
        """List threads owned by user_id, optionally filtered by title or message content."""
        uid = (user_id or "").strip()
        if not uid:
            raise MemoryPersistenceError("user_id is required to list threads.")
        cleaned_search = (search_query or "").strip()
        with self._lock:
            try:
                assert self._conn is not None
                if cleaned_search:
                    like_term = f"%{cleaned_search.lower()}%"
                    rows = self._conn.execute(
                        """
                        SELECT t.thread_id, t.title, t.chat_mode, t.is_pinned,
                               t.created_at, t.updated_at,
                               COUNT(m.id) AS message_count
                        FROM threads t
                        LEFT JOIN thread_messages m
                          ON m.thread_id = t.thread_id AND m.user_id = t.user_id
                        WHERE t.user_id = ?
                          AND (
                            LOWER(t.title) LIKE ?
                            OR EXISTS (
                                SELECT 1 FROM thread_messages sm
                                WHERE sm.thread_id = t.thread_id
                                  AND sm.user_id = t.user_id
                                  AND LOWER(sm.content) LIKE ?
                            )
                          )
                        GROUP BY t.thread_id
                        ORDER BY t.is_pinned DESC, t.updated_at DESC
                        """,
                        (uid, like_term, like_term),
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        """
                        SELECT t.thread_id, t.title, t.chat_mode, t.is_pinned,
                               t.created_at, t.updated_at,
                               COUNT(m.id) AS message_count
                        FROM threads t
                        LEFT JOIN thread_messages m
                          ON m.thread_id = t.thread_id AND m.user_id = t.user_id
                        WHERE t.user_id = ?
                        GROUP BY t.thread_id
                        ORDER BY t.is_pinned DESC, t.updated_at DESC
                        """,
                        (uid,),
                    ).fetchall()

                now_ts = time.time()
                return [
                    {
                        "id": row["thread_id"],
                        "thread_id": row["thread_id"],
                        "title": row["title"],
                        "chat_mode": row["chat_mode"],
                        "is_pinned": bool(row["is_pinned"]),
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                        "message_count": row["message_count"],
                        "bucket": classify_time_bucket(row["updated_at"], now_ts=now_ts),
                        "timestamp": time.strftime(
                            "%b %d, %H:%M",
                            time.localtime(row["updated_at"]),
                        ),
                    }
                    for row in rows
                ]
            except Exception as exc:
                raise MemoryPersistenceError(f"Failed retrieving threads: {exc}") from exc

    def retrieve_chat_history(
        self,
        thread_id: str,
        user_id: str,
        graph: Any = None,
    ) -> List[Dict[str, Any]]:
        """Restore persisted messages for thread_id after verifying user ownership."""
        tid, uid = self._validate_ids(thread_id, user_id)
        with self._lock:
            try:
                assert self._conn is not None
                self._verify_thread_ownership(tid, uid)
                rows = self._conn.execute(
                    """
                    SELECT role, content, sources_json, web_sources_json,
                           tool_events_json, image_b64, image_mime, timestamp
                    FROM thread_messages
                    WHERE thread_id = ? AND user_id = ?
                    ORDER BY id ASC
                    """,
                    (tid, uid),
                ).fetchall()
                if rows:
                    return [
                        {
                            "role": row["role"],
                            "content": row["content"],
                            "sources": json.loads(row["sources_json"] or "[]"),
                            "web_sources": json.loads(row["web_sources_json"] or "[]"),
                            "tool_events": json.loads(row["tool_events_json"] or "[]"),
                            "image_b64": row["image_b64"],
                            "image_mime": row["image_mime"],
                            "timestamp": row["timestamp"],
                        }
                        for row in rows
                    ]

                if graph is not None:
                    snapshot = graph.get_state({"configurable": {"thread_id": tid}})
                    values = getattr(snapshot, "values", {}) or {}
                    lc_msgs = values.get("messages") or []
                    restored: List[Dict[str, Any]] = []
                    for msg in lc_msgs:
                        msg_type = getattr(msg, "type", "")
                        if msg_type == "human":
                            restored.append({"role": "user", "content": str(msg.content)})
                        elif msg_type == "ai" and str(msg.content).strip():
                            restored.append({"role": "assistant", "content": str(msg.content)})
                    return restored
                return []
            except MemoryPersistenceError:
                raise
            except Exception as exc:
                raise MemoryPersistenceError(f"Failed retrieving chat history: {exc}") from exc

    def delete_thread(self, thread_id: str, user_id: str) -> bool:
        """Delete a conversation thread and its checkpoints after verifying ownership."""
        tid, uid = self._validate_ids(thread_id, user_id)
        with self._lock:
            try:
                assert self._conn is not None
                self._verify_thread_ownership(tid, uid)
                cur = self._conn.execute(
                    "DELETE FROM threads WHERE thread_id = ? AND user_id = ?",
                    (tid, uid),
                )
                self._conn.execute(
                    "DELETE FROM thread_messages WHERE thread_id = ? AND user_id = ?",
                    (tid, uid),
                )
                self._conn.execute(
                    "DELETE FROM checkpoints WHERE thread_id = ?",
                    (tid,),
                )
                self._conn.execute(
                    "DELETE FROM writes WHERE thread_id = ?",
                    (tid,),
                )
                self._conn.commit()
                return cur.rowcount > 0
            except MemoryPersistenceError:
                raise
            except Exception as exc:
                raise MemoryPersistenceError(f"Failed deleting thread: {exc}") from exc


# ── Module-level convenience functions ────────────────────────────────────────

def create_new_thread(
    user_id: str,
    memory: Optional[ConversationMemoryManager] = None,
    title: str = "New Conversation",
    chat_mode: str = "Auto (Agentic)",
) -> str:
    manager = memory or ConversationMemoryManager()
    return manager.create_new_thread(user_id=user_id, title=title, chat_mode=chat_mode)


def retrieve_all_threads(
    user_id: str,
    memory: Optional[ConversationMemoryManager] = None,
    search_query: str = "",
) -> List[Dict[str, Any]]:
    manager = memory or ConversationMemoryManager()
    return manager.retrieve_all_threads(user_id=user_id, search_query=search_query)


def retrieve_chat_history(
    thread_id: str,
    user_id: str,
    memory: Optional[ConversationMemoryManager] = None,
    graph: Any = None,
) -> List[Dict[str, Any]]:
    manager = memory or ConversationMemoryManager()
    return manager.retrieve_chat_history(thread_id=thread_id, user_id=user_id, graph=graph)
