"""Service layer for user-scoped document ingestion and vector index management."""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from qna_bot.config.settings import AppSettings, get_settings
from qna_bot.rag.ingestion import DocumentIngestor, IngestionResult
from qna_bot.rag.vector_store import FAISSVectorStoreManager


class DocumentService:
    """High-level API for uploading, listing, removing, and clearing user documents."""

    def __init__(
        self,
        settings: Optional[AppSettings] = None,
        vector_store: Optional[FAISSVectorStoreManager] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.vector_store = vector_store or FAISSVectorStoreManager(
            base_dir=self.settings.vector_store_dir
        )
        self.ingestor = DocumentIngestor(
            vector_store=self.vector_store,
            settings=self.settings,
        )

    def ingest_files(
        self,
        files: Sequence[Tuple[str, bytes]],
        user_id: str,
    ) -> List[IngestionResult]:
        """Validate, chunk, embed, and index uploaded documents for user_id."""
        return self.ingestor.ingest_batch(files=files, user_id=user_id)

    def list_documents(self, user_id: str) -> List[Dict[str, Any]]:
        """List indexed document records owned by user_id."""
        return self.vector_store.list_documents(user_id=user_id)

    def remove_document(self, document_id: str, user_id: str) -> bool:
        """Remove a specific document from user_id's vector store."""
        return self.vector_store.delete_document(document_id=document_id, user_id=user_id)

    def clear_index(self, user_id: str) -> None:
        """Clear all indexed documents belonging to user_id."""
        self.vector_store.clear(user_id=user_id)
