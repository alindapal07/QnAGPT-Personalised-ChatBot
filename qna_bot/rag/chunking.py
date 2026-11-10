"""Text cleaning and chunk splitting with source metadata preservation."""

import re
from typing import List, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text: str) -> str:
    """Normalize line endings, null characters, and excessive blank lines."""
    if not text:
        return ""
    cleaned = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def split_documents(
    documents: Sequence[Document],
    document_id: str,
    user_id: str,
    file_hash: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """Split loaded document pages/sections into metadata-rich vector chunks."""
    cleaned_docs: List[Document] = []
    for doc in documents:
        text = clean_text(doc.page_content)
        if not text:
            continue
        cleaned_docs.append(Document(page_content=text, metadata=dict(doc.metadata)))

    if not cleaned_docs:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(cleaned_docs)

    enriched: List[Document] = []
    for idx, chunk in enumerate(chunks):
        text = chunk.page_content.strip()
        if not text:
            continue
        meta = dict(chunk.metadata)
        meta.update(
            {
                "document_id": document_id,
                "user_id": user_id,
                "file_hash": file_hash,
                "chunk_index": idx,
                "chunk_id": f"{document_id}:{idx}",
            }
        )
        enriched.append(Document(page_content=text, metadata=meta))

    return enriched
