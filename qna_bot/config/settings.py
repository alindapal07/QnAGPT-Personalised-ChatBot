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
    page_title: str = "AI Assistant"
    page_icon: str = "square"
    log_level: str = "INFO"
    stream_chunk_delay_sec: float = 0.02


def load_settings() -> AppSettings:
    """Load environment files and construct a fresh AppSettings instance."""
    _load_environment_files()
    return AppSettings(
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        log_level=os.getenv("LOG_LEVEL", "INFO").strip() or "INFO",
    )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return a cached AppSettings singleton for application runtime."""
    return load_settings()
