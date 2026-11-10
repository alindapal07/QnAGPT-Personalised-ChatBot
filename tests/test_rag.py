"""Pytest suite for RAG loaders, chunking, embeddings, FAISS vector store, ingestion, retrieval, and grading."""

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import zipfile

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import pytest

from qna_bot.config.settings import AppSettings
from qna_bot.core.exceptions import (
    DocumentIngestionError,
    EmbeddingError,
    VectorStoreError,
)
from qna_bot.rag.chunking import clean_text, split_documents
from qna_bot.rag.embeddings import get_embeddings, reset_embeddings_cache
from qna_bot.rag.grader import DocumentGrader, _BatchGradeSchema
from qna_bot.rag.ingestion import DocumentIngestor
from qna_bot.rag.loaders import load_document_bytes
from qna_bot.rag.retriever import DocumentRetriever
from qna_bot.rag.vector_store import FAISSVectorStoreManager, RetrievedChunk


class DeterministicTestEmbeddings(Embeddings):
    """Lightweight deterministic 8-dim normalized embeddings for offline FAISS unit tests."""

    def _encode(self, text: str) -> list[float]:
        lowered = text.lower()
        buckets = [0.0] * 8
        keywords = [
            "supervised",
            "neural",
            "faiss",
            "langchain",
            "python",
            "weather",
            "finance",
            "report",
        ]
        for idx, kw in enumerate(keywords):
            if kw in lowered:
                buckets[idx] += 2.0
        digest = hashlib.sha256(lowered.encode("utf-8")).digest()
        for i in range(8):
            buckets[i] += digest[i] / 255.0
        norm = sum(v * v for v in buckets) ** 0.5 or 1.0
        return [v / norm for v in buckets]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._encode(text)


def _make_pdf_bytes(text: str) -> bytes:
    safe_text = text.replace("(", "").replace(")", "")
    obj1 = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    obj2 = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    obj3 = (
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    )
    stream = f"BT /F1 12 Tf 72 720 Td ({safe_text}) Tj ET".encode("latin-1")
    obj4 = (
        f"4 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode("ascii")
        + stream
        + b"\nendstream\nendobj\n"
    )
    obj5 = b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    body = b"%PDF-1.4\n"
    offsets = []
    for obj in (obj1, obj2, obj3, obj4, obj5):
        offsets.append(len(body))
        body += obj
    xref_pos = len(body)
    xref = b"xref\n0 6\n0000000000 65535 f \n" + b"".join(
        f"{off:010d} 00000 n \n".encode("ascii") for off in offsets
    )
    return body + xref + f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("ascii")


def _make_docx_bytes(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>",
        )
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="word/document.xml"/>'
            "</Relationships>",
        )
        zf.writestr(
            "word/document.xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>",
        )
    return buf.getvalue()


def test_pdf_txt_docx_csv_loaders_preserve_metadata() -> None:
    pdf_docs = load_document_bytes(_make_pdf_bytes("Supervised learning uses labeled data."), "ml_paper.pdf")
    assert len(pdf_docs) == 1
    assert "Supervised learning" in pdf_docs[0].page_content
    assert pdf_docs[0].metadata["source"] == "ml_paper.pdf"
    assert pdf_docs[0].metadata["page"] == 1

    txt_docs = load_document_bytes(b"Neural networks contain layers of weights.", "notes.txt")
    assert txt_docs[0].metadata["source"] == "notes.txt"
    assert "page" not in txt_docs[0].metadata

    docx_docs = load_document_bytes(_make_docx_bytes("Docx architectural notes on FAISS."), "arch.docx")
    assert "FAISS" in docx_docs[0].page_content
    assert docx_docs[0].metadata["source"] == "arch.docx"

    csv_docs = load_document_bytes(b"topic,summary\nRAG,Retrieval augmented generation\n", "metrics.csv")
    assert len(csv_docs) == 1
    assert "Retrieval augmented generation" in csv_docs[0].page_content
    assert csv_docs[0].metadata["source"] == "metrics.csv"


def test_loader_rejects_invalid_extension_empty_and_oversized() -> None:
    with pytest.raises(DocumentIngestionError):
        load_document_bytes(b"exec", "malware.exe")
    with pytest.raises(DocumentIngestionError):
        load_document_bytes(b"", "empty.txt")
    with pytest.raises(DocumentIngestionError):
        load_document_bytes(b"a" * (2 * 1024 * 1024), "big.txt", max_size_mb=1)


def test_text_splitting_preserves_source_and_page_metadata() -> None:
    raw = [
        Document(
            page_content=clean_text(("LangChain RAG pipeline chunking test. " * 60)),
            metadata={"source": "guide.pdf", "page": 3},
        )
    ]
    chunks = split_documents(
        raw,
        document_id="doc-1",
        user_id="alice",
        file_hash="hash-1",
        chunk_size=300,
        chunk_overlap=50,
    )
    assert len(chunks) > 2
    assert all(c.metadata["source"] == "guide.pdf" for c in chunks)
    assert all(c.metadata["page"] == 3 for c in chunks)
    assert all(c.metadata["user_id"] == "alice" for c in chunks)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_embeddings_lazy_singleton_and_failure_handling() -> None:
    reset_embeddings_cache()
    mock_instance = MagicMock()
    with patch("qna_bot.rag.embeddings.HuggingFaceEmbeddings", return_value=mock_instance) as mock_cls:
        first = get_embeddings("sentence-transformers/all-MiniLM-L6-v2")
        second = get_embeddings("sentence-transformers/all-MiniLM-L6-v2")
        assert first is second
        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["encode_kwargs"] == {"normalize_embeddings": True}

    reset_embeddings_cache()
    with patch("qna_bot.rag.embeddings.HuggingFaceEmbeddings", side_effect=OSError("offline")):
        with pytest.raises(EmbeddingError):
            get_embeddings("sentence-transformers/all-MiniLM-L6-v2")
    reset_embeddings_cache()


def test_faiss_persistence_duplicate_prevention_deletion_and_user_isolation(tmp_path: Path) -> None:
    settings = AppSettings(groq_api_key="", data_dir=tmp_path, vector_store_dir=tmp_path / "vs")
    store = FAISSVectorStoreManager(
        base_dir=settings.vector_store_dir,
        embeddings=DeterministicTestEmbeddings(),
    )
    ingestor = DocumentIngestor(vector_store=store, settings=settings)

    res1 = ingestor.ingest_bytes(
        _make_pdf_bytes("Supervised learning uses labeled datasets."),
        "ml.pdf",
        user_id="alice",
    )
    assert res1.status == "ingested"
    assert res1.chunk_count >= 1

    # Duplicate ingestion prevented
    dup = ingestor.ingest_bytes(
        _make_pdf_bytes("Supervised learning uses labeled datasets."),
        "ml_copy.pdf",
        user_id="alice",
    )
    assert dup.status == "duplicate"

    # Second document for alice
    res2 = ingestor.ingest_bytes(
        b"Python type hints improve maintainability.",
        "python.txt",
        user_id="alice",
    )
    assert res2.status == "ingested"

    # Persistence across fresh manager instance
    reloaded = FAISSVectorStoreManager(
        base_dir=settings.vector_store_dir,
        embeddings=DeterministicTestEmbeddings(),
    )
    alice_hits = reloaded.search_documents("supervised labeled datasets", user_id="alice", top_k=2)
    assert alice_hits
    assert alice_hits[0].source == "ml.pdf"
    assert alice_hits[0].page == 1

    # MMR search works cleanly
    mmr_hits = reloaded.search_documents("python", user_id="alice", top_k=2, use_mmr=True)
    assert mmr_hits

    # Isolation: Bob sees neither Alice's documents nor chunks
    assert reloaded.list_documents("bob") == []
    assert reloaded.search_documents("supervised", user_id="bob", top_k=4) == []
    assert reloaded.delete_document(res1.document_id, user_id="bob") is False

    # Delete one of Alice's documents while keeping the other
    assert reloaded.delete_document(res1.document_id, user_id="alice") is True
    remaining = reloaded.list_documents("alice")
    assert len(remaining) == 1
    assert remaining[0]["filename"] == "python.txt"


def test_faiss_rejects_corrupted_or_tampered_pickle(tmp_path: Path) -> None:
    store = FAISSVectorStoreManager(
        base_dir=tmp_path / "vs",
        embeddings=DeterministicTestEmbeddings(),
    )
    ingestor = DocumentIngestor(
        vector_store=store,
        settings=AppSettings(groq_api_key="", vector_store_dir=tmp_path / "vs"),
    )
    ingestor.ingest_bytes(b"Tamper check document.", "sec.txt", user_id="alice")

    pkl_path = tmp_path / "vs" / "alice" / "index.pkl"
    pkl_path.write_bytes(b"corrupted-or-tampered-bytes")

    with pytest.raises(VectorStoreError):
        store.load_vector_store("alice")

    # Missing index handled cleanly by DocumentRetriever
    retriever = DocumentRetriever(vector_store=store)
    assert retriever.retrieve("anything", user_id="nonexistent_user") == []


def test_document_grader_short_circuits_low_scores_and_filters_chunks() -> None:
    grader = DocumentGrader(api_key="gsk_test", score_threshold=0.35)

    # Empty retrieval
    empty_res = grader.grade("query", [])
    assert empty_res.classification == "IRRELEVANT"
    assert empty_res.used_llm is False

    # Low similarity score short-circuits LLM call
    low_chunks = [
        RetrievedChunk(
            content="unrelated",
            metadata={},
            score=0.1,
            document_id="d1",
            source="doc.txt",
            page=None,
        )
    ]
    low_res = grader.grade("query", low_chunks)
    assert low_res.classification == "IRRELEVANT"
    assert low_res.used_llm is False

    # Plausible scores invoke LLM structured output and remove irrelevant chunks
    good_chunks = [
        RetrievedChunk("FAISS stores vectors.", {}, 0.85, "d1", "rag.pdf", 2),
        RetrievedChunk("Banana bread recipe.", {}, 0.55, "d2", "food.txt", None),
    ]
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = _BatchGradeSchema(
        overall_relevance="RELEVANT",
        relevant_indices=[0],
        reason="Chunk 0 directly discusses FAISS.",
    )
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    with patch.object(grader, "_make_llm", return_value=mock_llm):
        graded = grader.grade("How does FAISS work?", good_chunks)

    assert graded.classification == "RELEVANT"
    assert graded.used_llm is True
    assert len(graded.relevant_chunks) == 1
    assert graded.relevant_chunks[0].source == "rag.pdf"
