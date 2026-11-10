"""SQLAlchemy 2.x ORM model for locally stored documents and Qdrant indexing metadata."""

from datetime import datetime
import uuid

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from qna_bot.models.base import Base, utc_now


class UploadedDocument(Base):
    """Persistent document lifecycle record linking LocalStorageProvider and Qdrant."""

    __tablename__ = "uploaded_documents"
    __table_args__ = (
        Index("idx_uploaded_docs_user_checksum", "user_id", "workspace_id", "checksum"),
        Index("idx_uploaded_docs_user_status", "user_id", "processing_status"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: f"doc_{uuid.uuid4().hex[:14]}")
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    workspace_id: Mapped[str] = mapped_column(String(64), nullable=False, default="default")
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False, default="")
    file_type: Mapped[str] = mapped_column(String(24), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    checksum: Mapped[str] = mapped_column(String(128), nullable=False)
    processing_status: Mapped[str] = mapped_column(String(32), nullable=False, default="uploaded")
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    page_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    qdrant_collection: Mapped[str] = mapped_column(String(120), nullable=False, default="qna_documents")
    error_message: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )
