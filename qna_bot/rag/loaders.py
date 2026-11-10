"""Document loaders with extension, size, and content validation."""

from pathlib import Path
import tempfile
from typing import List, Sequence

from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

from qna_bot.core.exceptions import DocumentIngestionError
from qna_bot.core.logging_config import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = ("pdf", "txt", "docx", "csv")


def validate_document_input(
    file_bytes: bytes,
    filename: str,
    max_size_mb: int = 20,
    allowed_types: Sequence[str] = SUPPORTED_EXTENSIONS,
) -> tuple[str, str]:
    """Validate uploaded document name, extension, and size."""
    clean_name = Path(filename).name.strip()
    if not clean_name:
        raise DocumentIngestionError("Uploaded document has an invalid filename.")

    ext = Path(clean_name).suffix.lstrip(".").lower()
    allowed = {item.lower() for item in allowed_types}
    if not ext or ext not in allowed:
        raise DocumentIngestionError(
            f"Unsupported document type '.{ext or 'unknown'}'. "
            f"Allowed types: {', '.join(allowed_types)}."
        )

    if not file_bytes:
        raise DocumentIngestionError(f"Document '{clean_name}' is empty.")

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise DocumentIngestionError(
            f"Document '{clean_name}' exceeds {max_size_mb}MB size limit."
        )

    return clean_name, ext


def _load_from_temp_file(temp_path: Path, ext: str) -> List[Document]:
    if ext == "pdf":
        return PyPDFLoader(str(temp_path)).load()
    if ext == "txt":
        return TextLoader(str(temp_path), encoding="utf-8", autodetect_encoding=True).load()
    if ext == "docx":
        return Docx2txtLoader(str(temp_path)).load()
    if ext == "csv":
        try:
            return CSVLoader(str(temp_path), encoding="utf-8").load()
        except UnicodeDecodeError:
            return CSVLoader(str(temp_path), encoding="latin-1").load()
    raise DocumentIngestionError(f"Unsupported extension: {ext}")


def load_document_bytes(
    file_bytes: bytes,
    filename: str,
    max_size_mb: int = 20,
    allowed_types: Sequence[str] = SUPPORTED_EXTENSIONS,
) -> List[Document]:
    """Load uploaded file bytes via temporary storage while preserving clean metadata."""
    clean_name, ext = validate_document_input(
        file_bytes=file_bytes,
        filename=filename,
        max_size_mb=max_size_mb,
        allowed_types=allowed_types,
    )

    try:
        with tempfile.TemporaryDirectory(prefix="qnabot_doc_") as tmp_dir:
            temp_path = Path(tmp_dir) / f"upload.{ext}"
            temp_path.write_bytes(file_bytes)
            raw_docs = _load_from_temp_file(temp_path, ext)
    except DocumentIngestionError:
        raise
    except Exception as exc:
        logger.warning("Failed to parse document '%s': %s", clean_name, exc)
        raise DocumentIngestionError(
            f"Could not extract content from '{clean_name}': {exc}"
        ) from exc

    normalized: List[Document] = []
    for doc in raw_docs:
        text = (doc.page_content or "").strip()
        if not text:
            continue
        meta = dict(doc.metadata or {})
        raw_page = meta.get("page")
        page_num = (int(raw_page) + 1) if ext == "pdf" and isinstance(raw_page, int) else None

        clean_meta = {
            "source": clean_name,
            "file_type": ext,
        }
        if page_num is not None:
            clean_meta["page"] = page_num
        if "row" in meta:
            clean_meta["row"] = meta["row"]

        normalized.append(Document(page_content=text, metadata=clean_meta))

    if not normalized:
        raise DocumentIngestionError(
            f"Document '{clean_name}' contains no extractable text."
        )

    logger.info(
        "Loaded document '%s' (%s) with %d non-empty sections",
        clean_name,
        ext,
        len(normalized),
    )
    return normalized
