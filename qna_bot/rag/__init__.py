"""RAG subsystem exports (loaders, chunking, embeddings, FAISS store, ingestion, retrieval, grading)."""

from qna_bot.rag.chunking import clean_text, split_documents
from qna_bot.rag.embeddings import get_embeddings, reset_embeddings_cache
from qna_bot.rag.grader import DocumentGrader, GradeLabel, GradingResult
from qna_bot.rag.ingestion import DocumentIngestor, IngestionResult
from qna_bot.rag.loaders import load_document_bytes, validate_document_input
from qna_bot.rag.retriever import DocumentRetriever
from qna_bot.rag.vector_store import (
    FAISSVectorStoreManager,
    RetrievedChunk,
    VectorStoreBackend,
    add_documents,
    create_vector_store,
    delete_document,
    load_vector_store,
    save_vector_store,
    search_documents,
)

__all__ = [
    "DocumentGrader",
    "DocumentIngestor",
    "DocumentRetriever",
    "FAISSVectorStoreManager",
    "GradeLabel",
    "GradingResult",
    "IngestionResult",
    "RetrievedChunk",
    "VectorStoreBackend",
    "add_documents",
    "clean_text",
    "create_vector_store",
    "delete_document",
    "get_embeddings",
    "load_document_bytes",
    "load_vector_store",
    "reset_embeddings_cache",
    "save_vector_store",
    "search_documents",
    "split_documents",
    "validate_document_input",
]
