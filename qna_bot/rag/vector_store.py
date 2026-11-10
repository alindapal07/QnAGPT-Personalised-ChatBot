"""Modular vector storage backend with user-scoped FAISS persistence."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from qna_bot.config.settings import get_settings
from qna_bot.core.exceptions import VectorStoreError
from qna_bot.core.logging_config import get_logger
from qna_bot.rag.embeddings import get_embeddings

logger = get_logger(__name__)

_SAFE_USER_RE = re.compile(r"[^a-zA-Z0-9_-]+")


@dataclass(frozen=True)
class RetrievedChunk:
    """Normalized retrieval result independent of underlying vector backend."""

    content: str
    metadata: Dict[str, Any]
    score: float
    document_id: str
    source: str
    page: Optional[int] = None


class VectorStoreBackend(ABC):
    """Abstract vector storage interface designed for backend portability (e.g., FAISS -> Qdrant)."""

    @abstractmethod
    def has_document_hash(self, file_hash: str, user_id: str) -> bool:
        """Return True if a document with the given SHA-256 hash is already indexed for user_id."""

    @abstractmethod
    def add_documents(
        self,
        documents: Sequence[Document],
        user_id: str,
        document_meta: Mapping[str, Any],
    ) -> List[str]:
        """Index chunk documents under the given user namespace and return assigned chunk IDs."""

    @abstractmethod
    def delete_document(self, document_id: str, user_id: str) -> bool:
        """Delete a document owned by user_id from the index."""

    @abstractmethod
    def search_documents(
        self,
        query: str,
        user_id: str,
        top_k: int = 4,
        use_mmr: bool = False,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """Search indexed chunks strictly scoped to user_id."""

    @abstractmethod
    def list_documents(self, user_id: str) -> List[Dict[str, Any]]:
        """List all indexed document metadata records owned by user_id."""

    @abstractmethod
    def clear(self, user_id: str) -> None:
        """Remove all indexed documents and persisted files for user_id."""


def _sanitize_user_id(user_id: str) -> str:
    cleaned = _SAFE_USER_RE.sub("_", (user_id or "").strip()).strip("_")
    if not cleaned:
        raise VectorStoreError("A valid non-empty user_id is required for vector storage.")
    return cleaned


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


class FAISSVectorStoreManager(VectorStoreBackend):
    """User-isolated local FAISS vector store with manifest integrity verification."""

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        embeddings: Optional[Embeddings] = None,
        embeddings_provider: Optional[Callable[[], Embeddings]] = None,
    ) -> None:
        settings = get_settings()
        self.base_dir = Path(base_dir or settings.vector_store_dir).resolve()
        self._embeddings = embeddings
        self._embeddings_provider = embeddings_provider or (
            lambda: get_embeddings(settings.embedding_model_name)
        )
        self._lock = threading.RLock()

    def _get_embeddings(self) -> Embeddings:
        if self._embeddings is None:
            self._embeddings = self._embeddings_provider()
        return self._embeddings

    def _user_dir(self, user_id: str) -> Path:
        safe_id = _sanitize_user_id(user_id)
        candidate = (self.base_dir / safe_id).resolve()
        if candidate != self.base_dir and self.base_dir not in candidate.parents:
            raise VectorStoreError("Invalid vector store namespace path.")
        return candidate

    def _manifest_path(self, user_dir: Path) -> Path:
        return user_dir / "manifest.json"

    def _read_manifest(self, user_id: str) -> Dict[str, Any]:
        user_dir = self._user_dir(user_id)
        manifest_path = self._manifest_path(user_dir)
        if not manifest_path.is_file():
            return {"version": 1, "user_id": user_id, "checksums": {}, "documents": {}}
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise VectorStoreError(f"Corrupted vector store manifest: {exc}") from exc

        if data.get("user_id") != user_id:
            raise VectorStoreError("Vector store namespace ownership mismatch.")
        return data

    def _write_manifest(self, user_dir: Path, manifest: Dict[str, Any]) -> None:
        user_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path(user_dir).write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

    def _verify_index_integrity(self, user_dir: Path, manifest: Dict[str, Any]) -> None:
        """Verify local FAISS files match recorded manifest SHA-256 before loading pickle."""
        checksums = manifest.get("checksums") or {}
        for filename in ("index.faiss", "index.pkl"):
            file_path = (user_dir / filename).resolve()
            if user_dir not in file_path.parents or not file_path.is_file():
                raise VectorStoreError(f"Missing vector index artifact '{filename}'.")
            expected_hash = checksums.get(filename)
            if not expected_hash:
                raise VectorStoreError(
                    f"Refusing to load unverified FAISS artifact '{filename}' without checksum."
                )
            actual_hash = _sha256_file(file_path)
            if actual_hash != expected_hash:
                raise VectorStoreError(
                    f"Corrupted or tampered vector index artifact '{filename}'."
                )

    def create_vector_store(
        self,
        documents: Sequence[Document],
        user_id: str,
        ids: Optional[Sequence[str]] = None,
    ) -> FAISS:
        """Create a new FAISS index from documents scoped to user_id."""
        if not documents:
            raise VectorStoreError("Cannot create vector store from empty document list.")
        for doc in documents:
            if doc.metadata.get("user_id") != user_id:
                raise VectorStoreError("Document metadata user_id mismatch during indexing.")

        try:
            return FAISS.from_documents(
                list(documents),
                self._get_embeddings(),
                ids=list(ids) if ids else None,
            )
        except Exception as exc:
            logger.exception("Failed to create FAISS index for user=%s", user_id)
            raise VectorStoreError(f"Failed to build FAISS vector store: {exc}") from exc

    def load_vector_store(self, user_id: str) -> Optional[FAISS]:
        """Load and verify a persisted FAISS index for user_id, or return None if uninitialized."""
        with self._lock:
            user_dir = self._user_dir(user_id)
            index_file = user_dir / "index.faiss"
            pkl_file = user_dir / "index.pkl"
            manifest_file = self._manifest_path(user_dir)

            if not index_file.exists() and not pkl_file.exists() and not manifest_file.exists():
                return None

            manifest = self._read_manifest(user_id)
            if not manifest.get("documents"):
                return None

            self._verify_index_integrity(user_dir, manifest)

            try:
                return FAISS.load_local(
                    str(user_dir),
                    self._get_embeddings(),
                    allow_dangerous_deserialization=True,
                )
            except Exception as exc:
                logger.exception("Failed to load FAISS index for user=%s", user_id)
                raise VectorStoreError(f"Failed to load corrupted FAISS index: {exc}") from exc

    def save_vector_store(
        self,
        store: FAISS,
        user_id: str,
        manifest: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist a FAISS index and record SHA-256 artifact checksums."""
        with self._lock:
            user_dir = self._user_dir(user_id)
            user_dir.mkdir(parents=True, exist_ok=True)
            try:
                store.save_local(str(user_dir))
            except Exception as exc:
                logger.exception("Failed to save FAISS index for user=%s", user_id)
                raise VectorStoreError(f"Failed to save FAISS index: {exc}") from exc

            current_manifest = manifest if manifest is not None else self._read_manifest(user_id)
            current_manifest["user_id"] = user_id
            current_manifest["checksums"] = {
                "index.faiss": _sha256_file(user_dir / "index.faiss"),
                "index.pkl": _sha256_file(user_dir / "index.pkl"),
            }
            self._write_manifest(user_dir, current_manifest)

    def has_document_hash(self, file_hash: str, user_id: str) -> bool:
        with self._lock:
            manifest = self._read_manifest(user_id)
            return any(
                doc.get("file_hash") == file_hash
                for doc in manifest.get("documents", {}).values()
            )

    def add_documents(
        self,
        documents: Sequence[Document],
        user_id: str,
        document_meta: Mapping[str, Any],
    ) -> List[str]:
        """Add chunk documents to user_id's FAISS index and persist."""
        if not documents:
            raise VectorStoreError("No document chunks provided for indexing.")

        with self._lock:
            doc_id = str(document_meta["document_id"])
            chunk_ids = [
                str(doc.metadata.get("chunk_id") or f"{doc_id}:{idx}")
                for idx, doc in enumerate(documents)
            ]
            for doc, cid in zip(documents, chunk_ids):
                doc.metadata["user_id"] = user_id
                doc.metadata["document_id"] = doc_id
                doc.metadata["chunk_id"] = cid

            manifest = self._read_manifest(user_id)
            existing_store = self.load_vector_store(user_id)

            if existing_store is None:
                store = self.create_vector_store(documents, user_id=user_id, ids=chunk_ids)
            else:
                try:
                    existing_store.add_documents(list(documents), ids=chunk_ids)
                    store = existing_store
                except Exception as exc:
                    logger.exception("Failed adding chunks to FAISS for user=%s", user_id)
                    raise VectorStoreError(f"Failed adding documents to index: {exc}") from exc

            manifest.setdefault("documents", {})[doc_id] = {
                "document_id": doc_id,
                "filename": document_meta["filename"],
                "file_hash": document_meta["file_hash"],
                "file_type": document_meta.get("file_type", ""),
                "chunk_count": len(chunk_ids),
                "page_count": document_meta.get("page_count"),
                "chunk_ids": chunk_ids,
                "user_id": user_id,
                "created_at": document_meta.get("created_at", time.time()),
            }
            self.save_vector_store(store, user_id=user_id, manifest=manifest)
            return chunk_ids

    def delete_document(self, document_id: str, user_id: str) -> bool:
        """Delete all chunks belonging to document_id after verifying ownership."""
        with self._lock:
            user_dir = self._user_dir(user_id)
            manifest = self._read_manifest(user_id)
            documents = manifest.get("documents", {})
            record = documents.get(document_id)
            if not record:
                return False
            if record.get("user_id") != user_id:
                raise VectorStoreError("Unauthorized attempt to delete another user's document.")

            chunk_ids = list(record.get("chunk_ids") or [])
            store = self.load_vector_store(user_id)
            documents.pop(document_id, None)

            if not documents:
                self.clear(user_id)
                return True

            if store is not None and chunk_ids:
                try:
                    store.delete(ids=chunk_ids)
                except Exception as exc:
                    logger.exception("Failed to delete document %s from FAISS", document_id)
                    raise VectorStoreError(f"Failed deleting document chunks: {exc}") from exc
                self.save_vector_store(store, user_id=user_id, manifest=manifest)
            else:
                self._write_manifest(user_dir, manifest)

            return True

    def search_documents(
        self,
        query: str,
        user_id: str,
        top_k: int = 4,
        use_mmr: bool = False,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """Run similarity or MMR search scoped strictly to user_id."""
        cleaned_query = (query or "").strip()
        if not cleaned_query or top_k <= 0:
            return []

        with self._lock:
            store = self.load_vector_store(user_id)
            if store is None:
                return []

            combined_filter: Dict[str, Any] = {"user_id": user_id}
            if metadata_filter:
                for key, val in metadata_filter.items():
                    if key != "user_id":
                        combined_filter[key] = val

            try:
                if use_mmr:
                    mmr_docs = store.max_marginal_relevance_search(
                        cleaned_query,
                        k=top_k,
                        fetch_k=max(top_k * 3, 12),
                        filter=combined_filter,
                    )
                    scored_lookup = {
                        doc.metadata.get("chunk_id"): max(0.0, min(1.0, 1.0 - float(dist) / 2.0))
                        for doc, dist in store.similarity_search_with_score(
                            cleaned_query,
                            k=max(top_k * 3, 12),
                            filter=combined_filter,
                        )
                    }
                    pairs = [
                        (doc, scored_lookup.get(doc.metadata.get("chunk_id"), 0.5))
                        for doc in mmr_docs
                    ]
                else:
                    raw_pairs = store.similarity_search_with_score(
                        cleaned_query,
                        k=top_k,
                        filter=combined_filter,
                    )
                    pairs = [
                        (doc, max(0.0, min(1.0, 1.0 - float(dist) / 2.0)))
                        for doc, dist in raw_pairs
                    ]
            except Exception as exc:
                logger.exception("FAISS search failed for user=%s", user_id)
                raise VectorStoreError(f"Vector search failed: {exc}") from exc

            results: List[RetrievedChunk] = []
            for doc, score in pairs:
                meta = dict(doc.metadata or {})
                if meta.get("user_id") != user_id:
                    continue
                results.append(
                    RetrievedChunk(
                        content=doc.page_content,
                        metadata=meta,
                        score=round(float(score), 4),
                        document_id=str(meta.get("document_id", "")),
                        source=str(meta.get("source", "Unknown")),
                        page=meta.get("page"),
                    )
                )
            return results

    def list_documents(self, user_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            manifest = self._read_manifest(user_id)
            docs = list(manifest.get("documents", {}).values())
            docs.sort(key=lambda d: float(d.get("created_at", 0.0)), reverse=True)
            return docs

    def clear(self, user_id: str) -> None:
        with self._lock:
            user_dir = self._user_dir(user_id)
            if user_dir.exists():
                shutil.rmtree(user_dir, ignore_errors=True)


# ── Module-level convenience API ──────────────────────────────────────────────

def create_vector_store(
    documents: Sequence[Document],
    user_id: str,
    base_dir: Optional[Path] = None,
    embeddings: Optional[Embeddings] = None,
) -> FAISS:
    return FAISSVectorStoreManager(base_dir=base_dir, embeddings=embeddings).create_vector_store(
        documents, user_id=user_id
    )


def load_vector_store(
    user_id: str,
    base_dir: Optional[Path] = None,
    embeddings: Optional[Embeddings] = None,
) -> Optional[FAISS]:
    return FAISSVectorStoreManager(base_dir=base_dir, embeddings=embeddings).load_vector_store(
        user_id=user_id
    )


def save_vector_store(
    store: FAISS,
    user_id: str,
    base_dir: Optional[Path] = None,
    embeddings: Optional[Embeddings] = None,
) -> None:
    FAISSVectorStoreManager(base_dir=base_dir, embeddings=embeddings).save_vector_store(
        store, user_id=user_id
    )


def add_documents(
    documents: Sequence[Document],
    user_id: str,
    document_meta: Mapping[str, Any],
    base_dir: Optional[Path] = None,
    embeddings: Optional[Embeddings] = None,
) -> List[str]:
    return FAISSVectorStoreManager(base_dir=base_dir, embeddings=embeddings).add_documents(
        documents, user_id=user_id, document_meta=document_meta
    )


def delete_document(
    document_id: str,
    user_id: str,
    base_dir: Optional[Path] = None,
    embeddings: Optional[Embeddings] = None,
) -> bool:
    return FAISSVectorStoreManager(base_dir=base_dir, embeddings=embeddings).delete_document(
        document_id=document_id, user_id=user_id
    )


def search_documents(
    query: str,
    user_id: str,
    top_k: int = 4,
    use_mmr: bool = False,
    metadata_filter: Optional[Dict[str, Any]] = None,
    base_dir: Optional[Path] = None,
    embeddings: Optional[Embeddings] = None,
) -> List[RetrievedChunk]:
    return FAISSVectorStoreManager(base_dir=base_dir, embeddings=embeddings).search_documents(
        query=query,
        user_id=user_id,
        top_k=top_k,
        use_mmr=use_mmr,
        metadata_filter=metadata_filter,
    )
