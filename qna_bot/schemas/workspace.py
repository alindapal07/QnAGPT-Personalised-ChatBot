"""Pydantic v2 DTOs for workspace settings, document metadata, and feedback."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class DocumentSummary(BaseModel):
    """Unified SQLite + Storage + Qdrant metadata view of a user document."""

    id: str
    user_id: str
    workspace_id: str = "default"
    filename: str
    storage_path: str = ""
    file_type: str
    file_size: int = 0
    checksum: str
    processing_status: str
    chunk_count: int = 0
    page_count: int = 0
    qdrant_collection: str = "qna_documents"
    error_message: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class SettingUpdate(BaseModel):
    """Validated user preference entry."""

    setting_key: str = Field(..., min_length=1, max_length=128)
    setting_value: str = Field(default="")


class FeedbackInput(BaseModel):
    """Validated user rating on an assistant response."""

    message_id: int
    rating: Literal["up", "down"]
    comment: str = Field(default="", max_length=2000)
