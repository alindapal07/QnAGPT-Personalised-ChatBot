"""Lazy singleton local Hugging Face embedding provider."""

import threading
from typing import Dict

from langchain_huggingface import HuggingFaceEmbeddings

from qna_bot.core.exceptions import EmbeddingError
from qna_bot.core.logging_config import get_logger

logger = get_logger(__name__)

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_LOCK = threading.Lock()
_EMBEDDING_CACHE: Dict[str, HuggingFaceEmbeddings] = {}


def _detect_device() -> str:
    """Use CUDA if available, otherwise fall back safely to CPU."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def get_embeddings(model_name: str = _DEFAULT_MODEL) -> HuggingFaceEmbeddings:
    """Return a lazily initialized, cached HuggingFaceEmbeddings instance."""
    with _LOCK:
        if model_name in _EMBEDDING_CACHE:
            return _EMBEDDING_CACHE[model_name]

        device = _detect_device()
        logger.info("Initializing HuggingFaceEmbeddings model=%s on device=%s", model_name, device)
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as exc:
            logger.exception("Failed to initialize embedding model '%s'", model_name)
            raise EmbeddingError(
                f"Failed to load embedding model '{model_name}': {exc}"
            ) from exc

        _EMBEDDING_CACHE[model_name] = embeddings
        return embeddings


def reset_embeddings_cache() -> None:
    """Clear cached embedding models (primarily for test isolation)."""
    with _LOCK:
        _EMBEDDING_CACHE.clear()
