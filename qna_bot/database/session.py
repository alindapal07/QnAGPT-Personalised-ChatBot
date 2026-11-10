"""SQLAlchemy 2.x engine factory, SQLite PRAGMA hardening, and transactional session scopes."""

from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from qna_bot.core.logging_config import get_logger
from qna_bot.models.base import Base

logger = get_logger(__name__)


def _configure_sqlite_connection(dbapi_connection, _connection_record, is_memory: bool) -> None:
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys = ON;")
        cursor.execute("PRAGMA busy_timeout = 10000;")
        if not is_memory:
            cursor.execute("PRAGMA journal_mode = WAL;")
            cursor.execute("PRAGMA synchronous = NORMAL;")
    finally:
        cursor.close()


def create_sqlite_engine(db_url_or_path: Path | str) -> Engine:
    """Create a hardened SQLAlchemy 2.x Engine for SQLite file or :memory: databases."""
    raw = str(db_url_or_path or "").strip()
    is_memory = raw in (":memory:", "sqlite:///:memory:", "sqlite://")
    if is_memory:
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            future=True,
        )
    else:
        if raw.startswith("sqlite:///"):
            db_path = Path(raw.removeprefix("sqlite:///"))
            url = raw
        else:
            db_path = Path(raw)
            url = f"sqlite:///{db_path.resolve().as_posix()}"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(
            url,
            connect_args={"check_same_thread": False, "timeout": 30.0},
            future=True,
        )

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn, conn_record) -> None:
        _configure_sqlite_connection(dbapi_conn, conn_record, is_memory=is_memory)

    return engine


class SQLAlchemyDatabase:
    """Manages SQLAlchemy 2.x Engine lifecycle, schema creation, and transactional sessions."""

    def __init__(self, db_url_or_path: Optional[Path | str] = None, auto_create: bool = True) -> None:
        target = db_url_or_path if db_url_or_path is not None else Path("data/chatbot.db")
        self.target = str(target)
        self.engine = create_sqlite_engine(target)
        self.SessionFactory = sessionmaker(
            bind=self.engine,
            class_=Session,
            expire_on_commit=False,
            autoflush=False,
        )
        if auto_create:
            self.create_tables()

    def create_tables(self) -> None:
        """Create any missing tables cleanly using SQLAlchemy metadata."""
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide an atomic transactional SQLAlchemy Session."""
        session = self.SessionFactory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def close(self) -> None:
        """Dispose connection pool cleanly."""
        self.engine.dispose()
