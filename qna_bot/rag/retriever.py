"""Document retrieval service wrapping the modular vector store backend."""

from typing import Dict, List, Optional

from qna_bot.config.settings import AppSettings, get_settings
from qna_bot.core.exceptions import VectorStoreError
from qna_bot.core.logging_config import get_logger
from qna_bot.rag.vector_store import (
    FAISSVectorStoreManager,
    RetrievedChunk,
    VectorStoreBackend,
)

logger = get_logger(__name__)


class DocumentRetriever:
    """Retrieves top-k relevant document chunks strictly scoped to a user."""

    def __init__(
        self,
        vector_store: Optional[VectorStoreBackend] = None,
        settings: Optional[AppSettings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.vector_store = vector_store or FAISSVectorStoreManager(
            base_dir=self.settings.vector_store_dir
        )

    def retrieve(
        self,
        query: str,
        user_id: str,
        top_k: Optional[int] = None,
        use_mmr: bool = False,
        metadata_filter: Optional[Dict[str, object]] = None,
    ) -> List[RetrievedChunk]:
        """Retrieve relevant document chunks for the given query and user."""
        resolved_top_k = top_k if top_k is not None else self.settings.retrieval_top_k
        try:
            chunks = self.vector_store.search_documents(
                query=query,
                user_id=user_id,
                top_k=resolved_top_k,
                use_mmr=use_mmr,
                metadata_filter=metadata_filter,
            )
            logger.info(
                "Retrieved %d chunks for user=%s (top_k=%d, mmr=%s)",
                len(chunks),
                user_id,
                resolved_top_k,
                use_mmr,
            )
            return chunks
        except VectorStoreError:
            raise
        except Exception as exc:
            logger.exception("Unexpected retrieval error for user=%s", user_id)
            raise VectorStoreError(f"Retrieval failed: {exc}") from exc
