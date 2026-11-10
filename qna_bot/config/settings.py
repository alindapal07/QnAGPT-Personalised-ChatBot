"""Application configuration and environment loading."""

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from typing import Dict, Tuple

from dotenv import dotenv_values

PACKAGE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_DIR.parent

_LAST_ENV_SIGNATURE: Tuple[Tuple[str, int], ...] = ()
_LOADED_DOTENV_VALUES: Dict[str, str] = {}

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise AI assistant. "
    "Respond using clear Markdown, properly formatted code blocks with language "
    "identifiers, and professional explanations."
)


def _candidate_env_paths() -> Tuple[Path, ...]:
    cwd_env = Path.cwd() / ".env"
    candidates = (cwd_env, PROJECT_ROOT / ".env", PACKAGE_DIR / ".env", PROJECT_ROOT.parent / ".env")
    seen: list[Path] = []
    for p in candidates:
        try:
            rp = p.resolve()
        except OSError:
            rp = p
        if rp not in seen:
            seen.append(rp)
    return tuple(seen)


def _current_env_signature() -> Tuple[Tuple[str, int], ...]:
    sig: list[Tuple[str, int]] = []
    for env_path in _candidate_env_paths():
        if env_path.is_file():
            try:
                sig.append((str(env_path), env_path.stat().st_mtime_ns))
            except OSError:
                pass
    return tuple(sig)


def _load_environment_files() -> None:
    """Load and hot-refresh .env files without overriding explicit process environment variables."""
    global _LAST_ENV_SIGNATURE
    sig = _current_env_signature()
    _LAST_ENV_SIGNATURE = sig

    # Reverse order so higher-priority paths overwrite lower-priority defaults
    merged: Dict[str, str] = {}
    for env_path in reversed(_candidate_env_paths()):
        if not env_path.is_file():
            continue
        parsed = dotenv_values(env_path)
        for key, val in parsed.items():
            if key and val is not None:
                merged[str(key)] = str(val)

    for key, val in merged.items():
        cur = os.environ.get(key)
        prev = _LOADED_DOTENV_VALUES.get(key)
        if cur is None or cur == "" or (prev is not None and cur == prev):
            os.environ[key] = val
        _LOADED_DOTENV_VALUES[key] = val


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
    storage_dir: Path = PROJECT_ROOT / "storage"
    vector_store_dir: Path = PROJECT_ROOT / "data" / "vector_store"
    sqlite_db_path: Path = PROJECT_ROOT / "data" / "chatbot.db"
    database_url: str = ""
    langgraph_checkpoint_db: Path = PROJECT_ROOT / "data" / "chatbot.db"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_document_size_mb: int = 20
    allowed_document_types: Tuple[str, ...] = ("pdf", "txt", "docx", "csv", "md")
    retrieval_top_k: int = 4
    retrieval_score_threshold: float = 0.2
    max_retrieval_retries: int = 2
    max_validation_retries: int = 1

    # Qdrant vector store settings
    qdrant_mode: str = "local"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "qna_documents"

    # Multi-mode, Agent & MCP defaults
    default_user_id: str = "local_user"
    chat_modes: Tuple[str, ...] = ("Auto (Agentic)", "Document Chat", "General Chat")
    max_agent_iterations: int = 10
    mcp_connect_timeout_sec: int = 15
    mcp_tool_timeout_sec: int = 60
    mcp_max_output_chars: int = 16000

    # Authentication & OAuth settings
    session_secret_key: str = ""
    session_ttl_hours: int = 168
    app_base_url: str = "http://localhost:8501"
    google_client_id: str = ""
    google_client_secret: str = ""
    github_client_id: str = ""
    github_client_secret: str = ""

    # MCP Marketplace Registry & Credential Vault defaults
    official_registry_url: str = "https://registry.modelcontextprotocol.io"
    glama_api_url: str = "https://glama.ai/api/mcp/v1"
    glama_api_key: str = ""
    credential_encryption_key: str = ""
    oauth_redirect_uri: str = "http://localhost:8501/oauth/callback"
    registry_cache_ttl_sec: int = 300

    @property
    def resolved_database_url(self) -> str:
        if self.database_url.strip():
            return self.database_url.strip()
        return f"sqlite:///{self.sqlite_db_path.resolve().as_posix()}"


def _read_config_value(key: str, default: str = "") -> str:
    val = os.getenv(key, "").strip()
    return val if val else (_read_streamlit_secret(key) or default)


def load_settings() -> AppSettings:
    """Load environment files / secrets and construct a fresh AppSettings instance."""
    _load_environment_files()
    api_key = _read_config_value("GROQ_API_KEY")
    raw_data_dir = os.getenv("QNAGPT_DATA_DIR", "").strip()
    data_dir = Path(raw_data_dir) if raw_data_dir else (PROJECT_ROOT / "data")
    raw_storage_dir = os.getenv("QNAGPT_STORAGE_DIR", "").strip()
    if raw_storage_dir:
        storage_dir = Path(raw_storage_dir)
    elif raw_data_dir:
        storage_dir = data_dir / "storage"
    else:
        storage_dir = PROJECT_ROOT / "storage"

    sqlite_db_path = data_dir / "chatbot.db"
    raw_checkpoint_db = os.getenv("LANGGRAPH_CHECKPOINT_DB", "").strip()
    checkpoint_db_path = Path(raw_checkpoint_db) if raw_checkpoint_db else sqlite_db_path

    glama_key = _read_config_value("GLAMA_API_KEY")
    cred_key = _read_config_value("QNAGPT_CREDENTIAL_KEY")
    session_secret = _read_config_value("SESSION_SECRET_KEY")

    try:
        cache_ttl = int(os.getenv("QNAGPT_REGISTRY_CACHE_TTL_SEC", "300"))
    except ValueError:
        cache_ttl = 300
    try:
        session_ttl_hours = max(1, int(os.getenv("SESSION_TTL_HOURS", "168")))
    except ValueError:
        session_ttl_hours = 168

    qdrant_url = _read_config_value("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = _read_config_value("QDRANT_API_KEY")
    raw_qdrant_mode = _read_config_value("QDRANT_MODE").lower()
    if raw_qdrant_mode in ("local", "cloud"):
        qdrant_mode = raw_qdrant_mode
    elif qdrant_api_key or "cloud.qdrant.io" in qdrant_url.lower():
        qdrant_mode = "cloud"
    else:
        qdrant_mode = "local"

    return AppSettings(
        groq_api_key=api_key,
        log_level=os.getenv("LOG_LEVEL", "INFO").strip() or "INFO",
        data_dir=data_dir,
        storage_dir=storage_dir,
        vector_store_dir=data_dir / "vector_store",
        sqlite_db_path=sqlite_db_path,
        database_url=os.getenv("DATABASE_URL", "").strip(),
        langgraph_checkpoint_db=checkpoint_db_path,
        qdrant_mode=qdrant_mode,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        qdrant_collection=_read_config_value("QDRANT_COLLECTION", "qna_documents"),
        session_secret_key=session_secret,
        session_ttl_hours=session_ttl_hours,
        app_base_url=_read_config_value("APP_BASE_URL", "http://localhost:8501"),
        google_client_id=_read_config_value("GOOGLE_CLIENT_ID"),
        google_client_secret=_read_config_value("GOOGLE_CLIENT_SECRET"),
        github_client_id=_read_config_value("GITHUB_CLIENT_ID"),
        github_client_secret=_read_config_value("GITHUB_CLIENT_SECRET"),
        official_registry_url=os.getenv(
            "MCP_REGISTRY_URL", "https://registry.modelcontextprotocol.io"
        ).strip()
        or "https://registry.modelcontextprotocol.io",
        glama_api_url=os.getenv("GLAMA_API_URL", "https://glama.ai/api/mcp/v1").strip()
        or "https://glama.ai/api/mcp/v1",
        glama_api_key=glama_key,
        credential_encryption_key=cred_key,
        oauth_redirect_uri=os.getenv(
            "QNAGPT_OAUTH_REDIRECT_URI", "http://localhost:8501/oauth/callback"
        ).strip()
        or "http://localhost:8501/oauth/callback",
        registry_cache_ttl_sec=max(30, cache_ttl),
    )


@lru_cache(maxsize=1)
def _get_settings_cached() -> AppSettings:
    return load_settings()


def get_settings() -> AppSettings:
    """Return a cached AppSettings singleton, refreshing automatically if `.env` changes."""
    if _current_env_signature() != _LAST_ENV_SIGNATURE:
        _get_settings_cached.cache_clear()
    return _get_settings_cached()


get_settings.cache_clear = _get_settings_cached.cache_clear  # type: ignore[attr-defined]
