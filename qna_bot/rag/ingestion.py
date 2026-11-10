"""End-to-end document ingestion pipeline (validation, loading, chunking, indexing)."""

from dataclasses import dataclass
import hashlib
import time
from typing import Literal, Optional, Sequence
import uuid

from qna_bot.config.settings import AppSettings, get_settings
from qna_bot.core.exceptions import DocumentIngestionError, EmbeddingError, VectorStoreError
from qna_bot.core.logging_config import get_logger
from qna_bot.rag.chunking import split_documents
from qna_bot.rag.loaders import load_document_bytes
from qna_bot.rag.vector_store import FAISSVectorStoreManager, VectorStoreBackend

logger = get_logger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    """Outcome statistics for a single ingested document."""

    document_id: str
    filename: str
    file_hash: str
    chunk_count: int
    page_count: Optional[int]
    status: Literal["ingested", "duplicate", "failed"]
    error: str = ""


class DocumentIngestor:
    """Coordinates document parsing, chunking, deduplication, and vector indexing."""

    def __init__(
        self,
        vector_store: Optional[VectorStoreBackend] = None,
        settings: Optional[AppSettings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.vector_store = vector_store or FAISSVectorStoreManager(
            base_dir=self.settings.vector_store_dir
        )

    def ingest_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        user_id: str,
    ) -> IngestionResult:
        """Ingest a single uploaded document into the user's scoped vector store."""
        file_hash = hashlib.sha256(file_bytes or b"").hexdigest()

        try:
            if file_bytes and self.vector_store.has_document_hash(file_hash, user_id=user_id):
                logger.info(
                    "Skipping duplicate document '%s' for user=%s",
                    filename,
                    user_id,
                )
                return IngestionResult(
                    document_id="",
                    filename=filename,
                    file_hash=file_hash,
                    chunk_count=0,
                    page_count=None,
                    status="duplicate",
                )

            raw_docs = load_document_bytes(
                file_bytes=file_bytes,
                filename=filename,
                max_size_mb=self.settings.max_document_size_mb,
                allowed_types=self.settings.allowed_document_types,
            )

            document_id = str(uuid.uuid4())
            chunks = split_documents(
                documents=raw_docs,
                document_id=document_id,
                user_id=user_id,
                file_hash=file_hash,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            if not chunks:
                raise DocumentIngestionError(
                    f"Document '{filename}' yielded zero valid text chunks."
                )

            pages = {
                doc.metadata["page"]
                for doc in raw_docs
                if isinstance(doc.metadata.get("page"), int)
            }
            page_count = len(pages) if pages else None
            clean_name = str(raw_docs[0].metadata.get("source", filename))
            file_type = str(raw_docs[0].metadata.get("file_type", ""))

            self.vector_store.add_documents(
                chunks,
                user_id=user_id,
                document_meta={
                    "document_id": document_id,
                    "filename": clean_name,
                    "file_hash": file_hash,
                    "file_type": file_type,
                    "page_count": page_count,
                    "created_at": time.time(),
                },
            )

            logger.info(
                "Ingested '%s' (doc_id=%s, chunks=%d, user=%s)",
                clean_name,
                document_id,
                len(chunks),
                user_id,
            )
            return IngestionResult(
                document_id=document_id,
                filename=clean_name,
                file_hash=file_hash,
                chunk_count=len(chunks),
                page_count=page_count,
                status="ingested",
            )
        except (DocumentIngestionError, EmbeddingError, VectorStoreError) as exc:
            logger.warning("Ingestion failed for '%s': %s", filename, exc)
            return IngestionResult(
                document_id="",
                filename=filename,
                file_hash=file_hash,
                chunk_count=0,
                page_count=None,
                status="failed",
                error=str(exc),
            )

    def ingest_batch(
        self,
        files: Sequence[tuple[str, bytes]],
        user_id: str,
    ) -> list[IngestionResult]:
        """Ingest multiple uploaded files sequentially and collect per-file results."""
        return [
            self.ingest_bytes(file_bytes=content, filename=name, user_id=user_id)
            for name, content in files
        ]
