"""Qdrant-backed multi-user VectorStoreBackend supporting QDRANT_MODE=local and QDRANT_MODE=cloud."""

import math
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence
import uuid

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models

from qna_bot.config.settings import AppSettings, get_settings
from qna_bot.core.exceptions import ConfigurationError, VectorStoreError
from qna_bot.core.logging_config import get_logger
from qna_bot.rag.embeddings import get_embeddings
from qna_bot.rag.vector_store import (
    FAISSVectorStoreManager,
    RetrievedChunk,
    VectorStoreBackend,
    _sanitize_user_id,
)

logger = get_logger(__name__)


def _cosine_sim(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class QdrantVectorStoreManager(VectorStoreBackend):
    """Multi-tenant Qdrant vector store enforcing strict user_id payload isolation on every query."""

    def __init__(
        self,
        settings: Optional[AppSettings] = None,
        client: Optional[QdrantClient] = None,
        qdrant_mode: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
        location: Optional[str] = None,
        embedding_factory: Optional[Callable[[], Embeddings]] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.mode = (qdrant_mode or self.settings.qdrant_mode or "local").strip().lower()
        self.url = (qdrant_url or self.settings.qdrant_url or "http://localhost:6333").strip()
        self.api_key = (qdrant_api_key if qdrant_api_key is not None else self.settings.qdrant_api_key).strip()
        self.collection_name = (collection_name or self.settings.qdrant_collection or "qna_documents").strip()
        self.location = location
        self._embedding_factory = embedding_factory or (
            lambda: get_embeddings(self.settings.embedding_model_name)
        )
        self._client: Optional[QdrantClient] = client
        self._collection_initialized = False

    def _get_embeddings(self) -> Embeddings:
        try:
            return self._embedding_factory()
        except Exception as exc:
            raise VectorStoreError(f"Unable to initialize embedding model: {exc}") from exc

    def _get_client(self) -> QdrantClient:
        if self._client is not None:
            return self._client

        if self.location:
            self._client = QdrantClient(location=self.location)
            return self._client

        if self.mode == "cloud":
            if not self.url or not self.api_key:
                raise ConfigurationError(
                    "Qdrant Cloud requires both QDRANT_URL and QDRANT_API_KEY in .env."
                )
            self._client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=10.0,
                check_compatibility=False,
            )
        else:
            if not self.url:
                raise ConfigurationError("QDRANT_URL is required when QDRANT_MODE=local.")
            self._client = QdrantClient(
                url=self.url,
                timeout=5.0,
                check_compatibility=False,
            )
        return self._client

    def ensure_collection(self) -> None:
        """Create Qdrant collection and keyword payload indexes idempotently."""
        if self._collection_initialized:
            return
        client = self._get_client()
        try:
            exists = client.collection_exists(self.collection_name)
            if not exists:
                probe = self._get_embeddings().embed_query("dimension probe")
                dim = len(probe)
                if dim <= 0:
                    raise VectorStoreError("Embedding model returned an empty vector.")
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
                )
            if not self.location:
                for field_name in ("user_id", "workspace_id", "document_id", "file_hash", "source"):
                    try:
                        client.create_payload_index(
                            collection_name=self.collection_name,
                            field_name=field_name,
                            field_schema=models.PayloadSchemaType.KEYWORD,
                        )
                    except Exception:
                        pass
            self._collection_initialized = True
        except (ConfigurationError, VectorStoreError):
            raise
        except Exception as exc:
            raise VectorStoreError(
                f"Could not connect or initialize Qdrant ({self.mode} @ {self.url}): {exc}"
            ) from exc

    def health_check(self) -> Dict[str, Any]:
        """Non-throwing fast connectivity probe for UI health banners."""
        try:
            if self._client is not None or self.location:
                client = self._get_client()
            elif self.mode == "cloud":
                if not self.url or not self.api_key:
                    raise ConfigurationError(
                        "Qdrant Cloud requires both QDRANT_URL and QDRANT_API_KEY in .env."
                    )
                client = QdrantClient(
                    url=self.url,
                    api_key=self.api_key,
                    timeout=1.5,
                    check_compatibility=False,
                )
            else:
                client = QdrantClient(
                    url=self.url,
                    timeout=1.0,
                    check_compatibility=False,
                )
            cols = client.get_collections()
            names = [c.name for c in getattr(cols, "collections", [])]
            return {
                "ok": True,
                "mode": "memory" if self.location == ":memory:" else self.mode,
                "url": self.location or self.url,
                "collection": self.collection_name,
                "collection_ready": self.collection_name in names,
                "detail": "Connected",
            }
        except Exception as exc:
            return {
                "ok": False,
                "mode": self.mode,
                "url": self.location or self.url,
                "collection": self.collection_name,
                "collection_ready": False,
                "detail": str(exc),
            }

    @staticmethod
    def _build_filter(
        user_id: str,
        metadata_filter: Optional[Mapping[str, Any]] = None,
    ) -> models.Filter:
        safe_user = _sanitize_user_id(user_id)
        conditions: List[models.Condition] = [
            models.FieldCondition(key="user_id", match=models.MatchValue(value=safe_user))
        ]
        if metadata_filter:
            for key, value in metadata_filter.items():
                if key == "user_id" or value is None or value == "":
                    continue
                conditions.append(
                    models.FieldCondition(key=str(key), match=models.MatchValue(value=value))
                )
        return models.Filter(must=conditions)

    def has_document_hash(
        self,
        file_hash: str,
        user_id: str,
        workspace_id: str = "default",
    ) -> bool:
        if not file_hash:
            return False
        self.ensure_collection()
        client = self._get_client()
        flt = self._build_filter(user_id, {"file_hash": file_hash, "workspace_id": workspace_id or "default"})
        try:
            points, _ = client.scroll(
                collection_name=self.collection_name,
                scroll_filter=flt,
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            return len(points) > 0
        except Exception as exc:
            raise VectorStoreError(f"Qdrant hash check failed: {exc}") from exc

    def add_documents(
        self,
        documents: Sequence[Document],
        user_id: str,
        document_meta: Mapping[str, Any],
    ) -> List[str]:
        safe_user = _sanitize_user_id(user_id)
        docs = [d for d in documents if (d.page_content or "").strip()]
        if not docs:
            raise VectorStoreError("No non-empty text chunks were provided for indexing.")

        self.ensure_collection()
        client = self._get_client()
        embeddings = self._get_embeddings()

        doc_id = str(document_meta.get("document_id") or f"doc_{uuid.uuid4().hex[:12]}")
        source = str(document_meta.get("source") or document_meta.get("filename") or "document")
        file_hash = str(document_meta.get("file_hash") or "")
        workspace_id = str(document_meta.get("workspace_id") or "default")
        file_type = str(document_meta.get("file_type") or Path(source).suffix.lstrip(".").lower())
        file_size = int(document_meta.get("file_size_bytes") or document_meta.get("file_size") or 0)
        now_ts = time.time()

        texts = [d.page_content for d in docs]
        try:
            vectors = embeddings.embed_documents(texts)
        except Exception as exc:
            raise VectorStoreError(f"Embedding generation failed before Qdrant upsert: {exc}") from exc

        points: List[models.PointStruct] = []
        point_ids: List[str] = []
        for idx, (doc, vec) in enumerate(zip(docs, vectors)):
            pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{safe_user}:{doc_id}:{idx}"))
            meta = dict(doc.metadata or {})
            payload = {
                "page_content": doc.page_content,
                "user_id": safe_user,
                "workspace_id": workspace_id,
                "document_id": doc_id,
                "source": source,
                "file_hash": file_hash,
                "file_type": file_type,
                "file_size_bytes": file_size,
                "chunk_index": int(meta.get("chunk_index", idx)),
                "page": meta.get("page"),
                "created_at": now_ts,
            }
            points.append(models.PointStruct(id=pid, vector=list(vec), payload=payload))
            point_ids.append(pid)

        try:
            for start in range(0, len(points), 128):
                client.upsert(
                    collection_name=self.collection_name,
                    points=points[start : start + 128],
                    wait=True,
                )
        except Exception as exc:
            raise VectorStoreError(f"Qdrant upsert failed for '{source}': {exc}") from exc

        return point_ids

    def delete_document(
        self,
        document_id: str,
        user_id: str,
        workspace_id: Optional[str] = None,
    ) -> bool:
        self.ensure_collection()
        client = self._get_client()
        flt_meta: Dict[str, Any] = {"document_id": document_id}
        if workspace_id:
            flt_meta["workspace_id"] = workspace_id
        flt = self._build_filter(user_id, flt_meta)
        try:
            existing, _ = client.scroll(
                collection_name=self.collection_name,
                scroll_filter=flt,
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            if not existing:
                return False
            client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=flt),
                wait=True,
            )
            return True
        except Exception as exc:
            raise VectorStoreError(f"Qdrant delete failed for '{document_id}': {exc}") from exc

    def search_documents(
        self,
        query: str,
        user_id: str,
        top_k: int = 4,
        use_mmr: bool = False,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            return []

        self.ensure_collection()
        client = self._get_client()
        flt = self._build_filter(user_id, metadata_filter)
        try:
            query_vec = list(self._get_embeddings().embed_query(cleaned_query))
            fetch_limit = max(top_k * 3, top_k) if use_mmr else max(1, top_k)
            response = client.query_points(
                collection_name=self.collection_name,
                query=query_vec,
                query_filter=flt,
                limit=fetch_limit,
                with_payload=True,
                with_vectors=use_mmr,
            )
            hits = list(getattr(response, "points", []) or [])
        except Exception as exc:
            raise VectorStoreError(f"Qdrant search failed: {exc}") from exc

        if not hits:
            return []

        if use_mmr and len(hits) > top_k:
            hits = self._apply_mmr(hits, query_vec, top_k=top_k)
        else:
            hits = hits[:top_k]

        results: List[RetrievedChunk] = []
        for hit in hits:
            payload = dict(getattr(hit, "payload", {}) or {})
            content = str(payload.pop("page_content", "") or "")
            raw_score = float(getattr(hit, "score", 0.0) or 0.0)
            bounded_score = max(0.0, min(1.0, raw_score))
            page_val = payload.get("page")
            results.append(
                RetrievedChunk(
                    content=content,
                    metadata=payload,
                    score=bounded_score,
                    document_id=str(payload.get("document_id", "")),
                    source=str(payload.get("source", "Unknown")),
                    page=int(page_val) if isinstance(page_val, (int, float)) else None,
                )
            )
        return results

    @staticmethod
    def _apply_mmr(hits: list[Any], query_vec: Sequence[float], top_k: int, lambda_mult: float = 0.6) -> list[Any]:
        selected: list[Any] = []
        remaining = list(hits)
        while remaining and len(selected) < top_k:
            best_idx = 0
            best_score = -1e9
            for idx, cand in enumerate(remaining):
                cvec = getattr(cand, "vector", None)
                rel = float(getattr(cand, "score", 0.0) or 0.0)
                if not isinstance(cvec, (list, tuple)) or not selected:
                    mmr_val = rel
                else:
                    max_sim = max(
                        _cosine_sim(cvec, getattr(s, "vector", []) or [])
                        for s in selected
                        if isinstance(getattr(s, "vector", None), (list, tuple))
                    )
                    mmr_val = lambda_mult * rel - (1.0 - lambda_mult) * max_sim
                if mmr_val > best_score:
                    best_score = mmr_val
                    best_idx = idx
            selected.append(remaining.pop(best_idx))
        return selected

    def list_documents(
        self,
        user_id: str,
        workspace_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        self.ensure_collection()
        client = self._get_client()
        flt = self._build_filter(user_id, {"workspace_id": workspace_id} if workspace_id else None)
        docs_by_id: Dict[str, Dict[str, Any]] = {}
        offset = None
        try:
            while True:
                points, offset = client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=flt,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for pt in points:
                    payload = dict(getattr(pt, "payload", {}) or {})
                    doc_id = str(payload.get("document_id") or "")
                    if not doc_id:
                        continue
                    entry = docs_by_id.setdefault(
                        doc_id,
                        {
                            "document_id": doc_id,
                            "source": str(payload.get("source") or "Unknown"),
                            "workspace_id": str(payload.get("workspace_id") or "default"),
                            "file_hash": str(payload.get("file_hash") or ""),
                            "file_type": str(payload.get("file_type") or ""),
                            "file_size_bytes": int(payload.get("file_size_bytes") or 0),
                            "chunk_count": 0,
                            "pages": set(),
                            "created_at": float(payload.get("created_at") or 0.0),
                        },
                    )
                    entry["chunk_count"] += 1
                    if payload.get("page") is not None:
                        entry["pages"].add(payload["page"])
                if offset is None:
                    break
        except Exception as exc:
            raise VectorStoreError(f"Failed listing Qdrant documents: {exc}") from exc

        rows: List[Dict[str, Any]] = []
        for entry in docs_by_id.values():
            pages = sorted(p for p in entry.pop("pages") if isinstance(p, (int, float)))
            entry["page_count"] = len(pages)
            rows.append(entry)
        rows.sort(key=lambda r: float(r.get("created_at", 0.0)), reverse=True)
        return rows

    def clear(self, user_id: str) -> None:
        self.ensure_collection()
        client = self._get_client()
        flt = self._build_filter(user_id)
        try:
            client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=flt),
                wait=True,
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed clearing Qdrant documents for user: {exc}") from exc


def migrate_faiss_user_to_qdrant(
    faiss_dir: Path,
    qdrant_store: QdrantVectorStoreManager,
    source_user_id: str,
    target_user_id: Optional[str] = None,
) -> Dict[str, int]:
    """Copy indexed chunks safely from a legacy local FAISS user store into Qdrant."""
    faiss_mgr = FAISSVectorStoreManager(base_dir=faiss_dir, embeddings=qdrant_store._get_embeddings())
    legacy_docs = faiss_mgr.list_documents(source_user_id)
    if not legacy_docs:
        return {"documents_migrated": 0, "chunks_migrated": 0}

    dest_user = target_user_id or source_user_id
    vs = faiss_mgr.load_vector_store(source_user_id)
    if vs is None:
        return {"documents_migrated": 0, "chunks_migrated": 0}

    grouped: Dict[str, List[Document]] = {}
    for doc in vs.docstore._dict.values():
        did = str(doc.metadata.get("document_id") or "")
        if did:
            grouped.setdefault(did, []).append(doc)

    docs_migrated = 0
    chunks_migrated = 0
    for meta in legacy_docs:
        did = str(meta.get("document_id") or "")
        fhash = str(meta.get("file_hash") or "")
        if not did or (fhash and qdrant_store.has_document_hash(fhash, dest_user)):
            continue
        chunk_docs = sorted(
            grouped.get(did, []),
            key=lambda d: int(d.metadata.get("chunk_index", 0)),
        )
        if not chunk_docs:
            continue
        qdrant_store.add_documents(
            documents=chunk_docs,
            user_id=dest_user,
            document_meta=meta,
        )
        docs_migrated += 1
        chunks_migrated += len(chunk_docs)

    return {"documents_migrated": docs_migrated, "chunks_migrated": chunks_migrated}
