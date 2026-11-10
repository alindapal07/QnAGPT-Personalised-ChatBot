"""User-isolated SQLite repository for uploaded documents and Qdrant indexing states."""

from datetime import datetime, timezone
from typing import List, Optional
import uuid

from sqlalchemy import delete, select

from qna_bot.database.session import SQLAlchemyDatabase
from qna_bot.models.document import UploadedDocument


class DocumentRepository:
    """Tracks document storage paths, checksums, and Qdrant indexing statuses per user."""

    def __init__(self, db: SQLAlchemyDatabase) -> None:
        self.db = db

    def create(
        self,
        user_id: str,
        workspace_id: str,
        filename: str,
        storage_path: str,
        file_type: str,
        file_size: int,
        checksum: str,
        processing_status: str = "uploaded",
        qdrant_collection: str = "qna_documents",
        document_id: Optional[str] = None,
    ) -> UploadedDocument:
        with self.db.session_scope() as session:
            rec = UploadedDocument(
                id=document_id or f"doc_{uuid.uuid4().hex[:14]}",
                user_id=user_id,
                workspace_id=workspace_id or "default",
                filename=filename,
                storage_path=storage_path,
                file_type=file_type,
                file_size=int(file_size),
                checksum=checksum,
                processing_status=processing_status,
                qdrant_collection=qdrant_collection,
            )
            session.add(rec)
            session.flush()
            session.refresh(rec)
            return rec

    def get(self, document_id: str, user_id: str) -> Optional[UploadedDocument]:
        with self.db.session_scope() as session:
            return session.scalar(
                select(UploadedDocument).where(
                    UploadedDocument.id == document_id,
                    UploadedDocument.user_id == user_id,
                )
            )

    def find_by_checksum(
        self,
        user_id: str,
        workspace_id: str,
        checksum: str,
    ) -> Optional[UploadedDocument]:
        with self.db.session_scope() as session:
            return session.scalar(
                select(UploadedDocument).where(
                    UploadedDocument.user_id == user_id,
                    UploadedDocument.workspace_id == (workspace_id or "default"),
                    UploadedDocument.checksum == checksum,
                )
            )

    def update_status(
        self,
        document_id: str,
        user_id: str,
        processing_status: str,
        chunk_count: Optional[int] = None,
        page_count: Optional[int] = None,
        error_message: str = "",
        qdrant_collection: Optional[str] = None,
    ) -> Optional[UploadedDocument]:
        with self.db.session_scope() as session:
            rec = session.scalar(
                select(UploadedDocument).where(
                    UploadedDocument.id == document_id,
                    UploadedDocument.user_id == user_id,
                )
            )
            if rec is None:
                return None
            rec.processing_status = processing_status
            if chunk_count is not None:
                rec.chunk_count = int(chunk_count)
            if page_count is not None:
                rec.page_count = int(page_count)
            if qdrant_collection is not None:
                rec.qdrant_collection = qdrant_collection
            rec.error_message = str(error_message or "")[:2000]
            rec.updated_at = datetime.now(timezone.utc)
            session.flush()
            session.refresh(rec)
            return rec

    def list_for_user(
        self,
        user_id: str,
        workspace_id: Optional[str] = None,
    ) -> List[UploadedDocument]:
        with self.db.session_scope() as session:
            stmt = select(UploadedDocument).where(UploadedDocument.user_id == user_id)
            if workspace_id:
                stmt = stmt.where(UploadedDocument.workspace_id == workspace_id)
            stmt = stmt.order_by(UploadedDocument.created_at.desc())
            return list(session.scalars(stmt).all())

    def delete(self, document_id: str, user_id: str) -> Optional[UploadedDocument]:
        with self.db.session_scope() as session:
            rec = session.scalar(
                select(UploadedDocument).where(
                    UploadedDocument.id == document_id,
                    UploadedDocument.user_id == user_id,
                )
            )
            if rec is None:
                return None
            session.execute(
                delete(UploadedDocument).where(
                    UploadedDocument.id == document_id,
                    UploadedDocument.user_id == user_id,
                )
            )
            return rec
