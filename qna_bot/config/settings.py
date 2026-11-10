"""Application configuration and environment loading."""

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv

PACKAGE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_DIR.parent

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise AI assistant. "
    "Respond using clear Markdown, properly formatted code blocks with language "
    "identifiers, and professional explanations."
)


def _load_environment_files() -> None:
    """Load .env from the working directory, package directory, or project root."""
    load_dotenv()
    candidate_paths = (
        PROJECT_ROOT / ".env",
        PACKAGE_DIR / ".env",
        PROJECT_ROOT.parent / ".env",
    )
    for env_path in candidate_paths:
        if os.getenv("GROQ_API_KEY"):
            break
        if env_path.is_file():
            load_dotenv(env_path)


def _read_streamlit_secret(key: str) -> str:
    """Safely read an optional Streamlit Cloud secret without requiring runtime context."""
    try:
        import streamlit as st

        secrets = getattr(st, "secrets", None)
        if secrets and key in secrets:
            return str(secrets[key]).strip()
    except Exception:
        return ""
    return ""


@dataclass(frozen=True)
class AppSettings:
    """Immutable application settings loaded from defaults and environment variables."""

    groq_api_key: str
    default_text_model: str = "openai/gpt-oss-20b"
    default_vision_model: str = "qwen/qwen3.8-27b"
    available_text_models: Tuple[str, ...] = (
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
    )
    default_temperature: float = 0.7
    default_max_tokens: int = 2048
    default_system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_image_size_mb: int = 10
    allowed_image_types: Tuple[str, ...] = ("png", "jpg", "jpeg", "webp")
    page_title: str = "QnAGPT Assistant"
    page_icon: str = "square"
    log_level: str = "INFO"
    stream_chunk_delay_sec: float = 0.02

    # RAG & Vector Store settings
    data_dir: Path = PROJECT_ROOT / "data"
    vector_store_dir: Path = PROJECT_ROOT / "data" / "vector_store"
    sqlite_db_path: Path = PROJECT_ROOT / "data" / "chatbot.db"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_document_size_mb: int = 20
    allowed_document_types: Tuple[str, ...] = ("pdf", "txt", "docx", "csv")
    retrieval_top_k: int = 4
    retrieval_score_threshold: float = 0.2
    max_retrieval_retries: int = 2
    max_validation_retries: int = 1

    # Multi-mode & isolation defaults
    default_user_id: str = "local_user"
    chat_modes: Tuple[str, ...] = ("Auto (Agentic)", "Document Chat", "General Chat")


def load_settings() -> AppSettings:
    """Load environment files / secrets and construct a fresh AppSettings instance."""
    _load_environment_files()
    api_key = os.getenv("GROQ_API_KEY", "").strip() or _read_streamlit_secret("GROQ_API_KEY")
    data_dir = Path(os.getenv("QNAGPT_DATA_DIR", str(PROJECT_ROOT / "data")))
    return AppSettings(
        groq_api_key=api_key,
        log_level=os.getenv("LOG_LEVEL", "INFO").strip() or "INFO",
        data_dir=data_dir,
        vector_store_dir=data_dir / "vector_store",
        sqlite_db_path=data_dir / "chatbot.db",
    )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return a cached AppSettings singleton for application runtime."""
    return load_settings()
