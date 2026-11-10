"""User-scoped LangChain document retrieval and summary tools."""

from typing import List

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from qna_bot.core.logging_config import get_logger
from qna_bot.rag.retriever import DocumentRetriever

logger = get_logger(__name__)


class DocumentSearchInput(BaseModel):
    """Typed input schema for user-scoped document search."""

    query: str = Field(..., min_length=1, description="Semantic query to search uploaded documents.")
    top_k: int = Field(default=4, ge=1, le=10, description="Maximum chunks to retrieve.")


class DocumentSummaryInput(BaseModel):
    """Typed input schema for inspecting indexed uploaded documents."""

    topic: str = Field(
        default="overview summary main points",
        description="Optional focus topic across uploaded documents.",
    )


def build_document_tools(
    retriever: DocumentRetriever,
    user_id: str,
) -> List[BaseTool]:
    """Build LangChain tools strictly bound to user_id on the server side."""

    @tool("retrieve_uploaded_documents", args_schema=DocumentSearchInput)
    def retrieve_uploaded_documents(query: str, top_k: int = 4) -> str:
        """Retrieve relevant passages and citations from the user's uploaded documents."""
        try:
            chunks = retriever.retrieve(query=query, user_id=user_id, top_k=top_k)
        except Exception as exc:
            logger.warning("Document retrieval tool failed: %s", exc)
            return f"Document retrieval error: {exc}"

        if not chunks:
            return "No matching passages were found in the uploaded documents."

        lines = []
        for idx, chunk in enumerate(chunks, start=1):
            page_str = f", Page {chunk.page}" if chunk.page is not None else ""
            lines.append(
                f"[{idx}] Source: {chunk.source}{page_str} (score={chunk.score:.2f})\n"
                f"{chunk.content}"
            )
        return "\n\n".join(lines)

    @tool("summarize_uploaded_documents", args_schema=DocumentSummaryInput)
    def summarize_uploaded_documents(
        topic: str = "overview summary main points",
    ) -> str:
        """List uploaded documents for this user and retrieve representative excerpts."""
        try:
            docs = retriever.vector_store.list_documents(user_id=user_id)
            if not docs:
                return "No documents are currently indexed for this workspace."
            doc_list = ", ".join(
                f"{d.get('filename')} ({d.get('chunk_count', 0)} chunks)" for d in docs
            )
            chunks = retriever.retrieve(query=topic, user_id=user_id, top_k=4, use_mmr=True)
            excerpts = "\n\n".join(
                f"- {c.source}"
                + (f" (Page {c.page})" if c.page is not None else "")
                + f": {c.content[:500]}"
                for c in chunks
            )
            return f"Indexed documents: {doc_list}\n\nRepresentative excerpts:\n{excerpts}"
        except Exception as exc:
            logger.warning("Document summary tool failed: %s", exc)
            return f"Document summary error: {exc}"

    return [retrieve_uploaded_documents, summarize_uploaded_documents]
