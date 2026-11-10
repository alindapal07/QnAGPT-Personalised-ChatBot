"""Service layer unifying LocalStorageProvider, SQLite uploaded_documents metadata, and Qdrant indexing."""

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import uuid

from qna_bot.config.settings import AppSettings, get_settings
from qna_bot.core.exceptions import DocumentIngestionError
from qna_bot.core.logging_config import get_logger
from qna_bot.database.session import SQLAlchemyDatabase
from qna_bot.rag.chunking import split_documents
from qna_bot.rag.ingestion import DocumentIngestor, IngestionResult
from qna_bot.rag.loaders import SUPPORTED_EXTENSIONS, load_document_bytes, validate_document_input
from qna_bot.rag.qdrant_store import QdrantVectorStoreManager
from qna_bot.rag.vector_store import FAISSVectorStoreManager, VectorStoreBackend
from qna_bot.repositories import AuditAndTelemetryRepository, DocumentRepository
from qna_bot.storage.base import StorageProvider
from qna_bot.storage.local import LocalStorageProvider

logger = get_logger(__name__)


class DocumentService:
    """High-level API coordinating local original storage, SQLite metadata, and Qdrant vector indexing."""

    def __init__(
        self,
        settings: Optional[AppSettings] = None,
        vector_store: Optional[VectorStoreBackend] = None,
        storage: Optional[StorageProvider] = None,
        db: Optional[SQLAlchemyDatabase] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.db = db
        self.storage = storage
        self.doc_repo = DocumentRepository(db) if db is not None else None
        self.audit_repo = AuditAndTelemetryRepository(db) if db is not None else None

        if vector_store is not None:
            self.vector_store: VectorStoreBackend = vector_store
        elif self.db is not None and self.storage is not None:
            self.vector_store = QdrantVectorStoreManager(settings=self.settings)
        else:
            self.vector_store = FAISSVectorStoreManager(base_dir=self.settings.vector_store_dir)

        self.ingestor = DocumentIngestor(
            vector_store=self.vector_store,
            settings=self.settings,
        )

    def _storage_and_repo_ready(self) -> bool:
        return self.storage is not None and self.doc_repo is not None

    def ingest_files(
        self,
        files: Sequence[Tuple[str, bytes]],
        user_id: str,
        workspace_id: str = "default",
    ) -> List[IngestionResult]:
        """Validate, durably store originals, chunk, embed, and index uploaded documents for user_id."""
        if not self._storage_and_repo_ready():
            return self.ingestor.ingest_batch(files=files, user_id=user_id)

        assert self.storage is not None
        assert self.doc_repo is not None

        results: List[IngestionResult] = []
        for filename, file_bytes in files:
            results.append(self._ingest_single_durable(filename, file_bytes, user_id, workspace_id))
        return results

    def _ingest_single_durable(
        self,
        filename: str,
        file_bytes: bytes,
        user_id: str,
        workspace_id: str = "default",
    ) -> IngestionResult:
        assert self.storage is not None
        assert self.doc_repo is not None

        try:
            clean_name, ext = validate_document_input(
                file_bytes=file_bytes,
                filename=filename,
                max_size_mb=self.settings.max_document_size_mb,
                allowed_types=self.settings.allowed_document_types or SUPPORTED_EXTENSIONS,
            )
            checksum = hashlib.sha256(file_bytes).hexdigest()

            existing = self.doc_repo.find_by_checksum(user_id, workspace_id, checksum)
            if existing is not None:
                if existing.processing_status == "indexed":
                    return IngestionResult(
                        document_id=existing.id,
                        filename=existing.filename,
                        file_hash=existing.checksum,
                        chunk_count=existing.chunk_count,
                        page_count=existing.page_count,
                        status="duplicate",
                    )
                if existing.processing_status == "failed" and existing.storage_path:
                    return self.retry_document(existing.id, user_id)

            stored = self.storage.save_file(
                user_id=user_id,
                bucket="documents",
                filename=clean_name,
                data=file_bytes,
            )
            doc_id = f"doc_{uuid.uuid4().hex[:14]}"
            collection_name = getattr(self.vector_store, "collection_name", self.settings.qdrant_collection)
            self.doc_repo.create(
                user_id=user_id,
                workspace_id=workspace_id,
                filename=clean_name,
                storage_path=stored.relative_path,
                file_type=ext,
                file_size=stored.size_bytes,
                checksum=stored.checksum,
                processing_status="processing",
                qdrant_collection=collection_name,
                document_id=doc_id,
            )
        except Exception as exc:
            return IngestionResult(
                document_id="",
                filename=Path(filename).name or filename,
                file_hash="",
                chunk_count=0,
                page_count=None,
                status="failed",
                error=str(exc),
            )

        return self._index_existing_record(
            doc_id=doc_id,
            user_id=user_id,
            workspace_id=workspace_id,
            filename=clean_name,
            file_bytes=file_bytes,
            checksum=checksum,
            ext=ext,
            file_size=len(file_bytes),
        )

    def _index_existing_record(
        self,
        doc_id: str,
        user_id: str,
        workspace_id: str,
        filename: str,
        file_bytes: bytes,
        checksum: str,
        ext: str,
        file_size: int,
    ) -> IngestionResult:
        assert self.doc_repo is not None
        try:
            raw_docs = load_document_bytes(
                file_bytes=file_bytes,
                filename=filename,
                max_size_mb=self.settings.max_document_size_mb,
                allowed_types=self.settings.allowed_document_types or SUPPORTED_EXTENSIONS,
            )
            chunks = split_documents(
                documents=raw_docs,
                document_id=doc_id,
                user_id=user_id,
                file_hash=checksum,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            pages = sorted({d.metadata["page"] for d in raw_docs if isinstance(d.metadata.get("page"), int)})
            self.vector_store.add_documents(
                documents=chunks,
                user_id=user_id,
                document_meta={
                    "document_id": doc_id,
                    "source": filename,
                    "workspace_id": workspace_id,
                    "file_hash": checksum,
                    "file_type": ext,
                    "file_size_bytes": file_size,
                    "chunk_count": len(chunks),
                    "page_count": len(pages),
                },
            )
            self.doc_repo.update_status(
                document_id=doc_id,
                user_id=user_id,
                processing_status="indexed",
                chunk_count=len(chunks),
                page_count=len(pages),
                error_message="",
            )
            if self.audit_repo is not None:
                self.audit_repo.log_audit(
                    user_id=user_id,
                    action="document.indexed",
                    resource_type="uploaded_document",
                    resource_id=doc_id,
                    metadata={"filename": filename, "chunks": len(chunks)},
                )
            return IngestionResult(
                document_id=doc_id,
                filename=filename,
                file_hash=checksum,
                chunk_count=len(chunks),
                page_count=len(pages),
                status="ingested",
            )
        except Exception as exc:
            logger.warning("Qdrant/vector indexing failed for '%s' (original file preserved): %s", filename, exc)
            self.doc_repo.update_status(
                document_id=doc_id,
                user_id=user_id,
                processing_status="failed",
                error_message=str(exc),
            )
            return IngestionResult(
                document_id=doc_id,
                filename=filename,
                file_hash=checksum,
                chunk_count=0,
                page_count=None,
                status="failed",
                error=f"Original file saved, but vector indexing failed: {exc}",
            )

    def retry_document(self, document_id: str, user_id: str) -> IngestionResult:
        """Retry vector indexing for a failed upload using the preserved local file."""
        if not self._storage_and_repo_ready():
            raise DocumentIngestionError("Local file storage is not enabled in this runtime.")
        assert self.storage is not None
        assert self.doc_repo is not None

        rec = self.doc_repo.get(document_id, user_id)
        if rec is None:
            return IngestionResult(
                document_id=document_id,
                filename="document",
                file_hash="",
                chunk_count=0,
                page_count=None,
                status="failed",
                error="Document record not found for your account.",
            )
        file_bytes = self.storage.read_file(user_id, rec.storage_path)
        self.doc_repo.update_status(document_id=rec.id, user_id=user_id, processing_status="processing")
        return self._index_existing_record(
            doc_id=rec.id,
            user_id=user_id,
            workspace_id=rec.workspace_id,
            filename=rec.filename,
            file_bytes=file_bytes,
            checksum=rec.checksum,
            ext=rec.file_type,
            file_size=rec.file_size,
        )

    def list_documents(self, user_id: str, workspace_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List document records combining SQLite lifecycle metadata and vector backend stats."""
        if self.doc_repo is not None:
            rows = self.doc_repo.list_for_user(user_id=user_id, workspace_id=workspace_id)
            return [
                {
                    "document_id": r.id,
                    "filename": r.filename,
                    "source": r.filename,
                    "workspace_id": r.workspace_id,
                    "file_type": r.file_type,
                    "file_size_bytes": r.file_size,
                    "storage_path": r.storage_path,
                    "processing_status": r.processing_status,
                    "error_message": r.error_message,
                    "chunk_count": r.chunk_count,
                    "page_count": r.page_count,
                    "file_hash": r.checksum,
                    "created_at": r.created_at.timestamp() if r.created_at else 0.0,
                }
                for r in rows
            ]
        try:
            return self.vector_store.list_documents(user_id=user_id)
        except Exception as exc:
            logger.warning("Vector store listing unavailable: %s", exc)
            return []

    def remove_document(self, document_id: str, user_id: str) -> bool:
        """Delete document vectors from Qdrant, SQLite metadata, and original file from LocalStorageProvider."""
        removed = False
        try:
            removed = self.vector_store.delete_document(document_id=document_id, user_id=user_id)
        except Exception as exc:
            logger.warning("Vector deletion error for doc=%s: %s", document_id, exc)

        if self.doc_repo is not None:
            deleted_rec = self.doc_repo.delete(document_id=document_id, user_id=user_id)
            if deleted_rec is not None:
                removed = True
                if self.storage is not None and deleted_rec.storage_path:
                    try:
                        self.storage.delete_file(user_id=user_id, relative_path=deleted_rec.storage_path)
                    except Exception as exc:
                        logger.warning("Local storage file removal failed: %s", exc)
                if self.audit_repo is not None:
                    self.audit_repo.log_audit(
                        user_id=user_id,
                        action="document.deleted",
                        resource_type="uploaded_document",
                        resource_id=document_id,
                        metadata={"filename": deleted_rec.filename},
                    )
        return removed

    def read_original_file(self, document_id: str, user_id: str) -> Optional[Tuple[str, bytes]]:
        """Read the user-owned raw document bytes safely from LocalStorageProvider."""
        if not self._storage_and_repo_ready():
            return None
        assert self.doc_repo is not None
        assert self.storage is not None
        rec = self.doc_repo.get(document_id, user_id)
        if rec is None or not rec.storage_path:
            return None
        return rec.filename, self.storage.read_file(user_id, rec.storage_path)

    def clear_index(self, user_id: str) -> None:
        """Clear all vector points, metadata rows, and stored files belonging to user_id."""
        try:
            self.vector_store.clear(user_id=user_id)
        except Exception as exc:
            logger.warning("Vector store clear warning for user=%s: %s", user_id, exc)
        if self.doc_repo is not None:
            for rec in self.doc_repo.list_for_user(user_id=user_id):
                self.doc_repo.delete(rec.id, user_id)
        if isinstance(self.storage, LocalStorageProvider):
            for rel in self.storage.list_files(user_id, "documents"):
                self.storage.delete_file(user_id, rel)

    def vector_health(self) -> Dict[str, Any]:
        """Return Qdrant connectivity & collection status if supported by backend."""
        fn = getattr(self.vector_store, "health_check", None)
        if callable(fn):
            return fn()
        return {"ok": True, "mode": "faiss", "url": str(self.settings.vector_store_dir), "detail": "Local FAISS"}
