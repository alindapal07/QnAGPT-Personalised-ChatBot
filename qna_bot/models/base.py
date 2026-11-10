"""SQLAlchemy 2.x DeclarativeBase and shared UTC timestamp helpers."""

from datetime import datetime, timezone

from sqlalchemy.orm import DeclarativeBase


def utc_now() -> datetime:
    """Return timezone-aware UTC datetime for ORM defaults."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Root declarative base for all SQLite SQLAlchemy 2.x ORM models."""
