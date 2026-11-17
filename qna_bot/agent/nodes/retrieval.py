"""Document retrieval node for LangGraph workflow."""

from typing import Any, Dict

from qna_bot.agent.nodes.base import BaseNodeSuite, logger
from qna_bot.agent.state import AgentState


class RetrievalNodesMixin:
    """Mixin containing document retrieval node."""

    def retrieve_documents(self: BaseNodeSuite, state: AgentState) -> Dict[str, Any]:
        """Retrieve top-k chunks from the user's scoped FAISS or Qdrant vector store."""
        query = (state.get("rewritten_query") or state.get("user_query") or "").strip()
        user_id = state.get("user_id") or self.settings.default_user_id
        retrieval_count = int(state.get("retrieval_attempt_count", 0)) + 1

        try:
            chunks = self.retriever.retrieve(
                query=query,
                user_id=user_id,
                top_k=self.settings.retrieval_top_k,
            )
            serialized = [
                {
                    "content": c.content,
                    "metadata": c.metadata,
                    "score": c.score,
                    "document_id": c.document_id,
                    "source": c.source,
                    "page": c.page,
                }
                for c in chunks
            ]
            return {
                "retrieved_documents": serialized,
                "retrieval_attempt_count": retrieval_count,
                "activity": "Searching uploaded documents...",
            }
        except Exception as exc:
            logger.warning("Retrieval node error: %s", exc)
            return {
                "retrieved_documents": [],
                "retrieval_attempt_count": retrieval_count,
                "error": f"Document retrieval failed: {exc}",
                "activity": "Searching uploaded documents...",
            }
