"""
config.py — Application configuration and environment loading.
All constants and environment variables live here.
"""
import os
from dotenv import load_dotenv

# Load .env looking in CWD, next to config.py, and parent folders
load_dotenv()
for env_path in [
    os.path.join(os.path.dirname(__file__), ".env"),
    os.path.join(os.path.dirname(__file__), "..", ".env"),
    os.path.join(os.path.dirname(__file__), "..", "..", ".env"),
    "langchain_01/.env",
    "../.env",
]:
    if not os.getenv("GROQ_API_KEY") and os.path.exists(env_path):
        load_dotenv(env_path)

# ── API ────────────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# ── Models ─────────────────────────────────────────────────────────────────────
DEFAULT_TEXT_MODEL   = "openai/gpt-oss-20b"
DEFAULT_VISION_MODEL = "qwen/qwen3.8-27b"
AVAILABLE_TEXT_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
]

# ── Defaults ───────────────────────────────────────────────────────────────────
DEFAULT_TEMPERATURE  = 0.7
DEFAULT_MAX_TOKENS   = 2048
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise AI assistant. "
    "Respond using clear Markdown, properly formatted code blocks with language "
    "identifiers, and professional explanations."
)

# ── Image ──────────────────────────────────────────────────────────────────────
MAX_IMAGE_SIZE_MB    = 10
ALLOWED_IMAGE_TYPES  = ["png", "jpg", "jpeg", "webp"]

# ── App ────────────────────────────────────────────────────────────────────────
PAGE_TITLE = "AI Assistant"
PAGE_ICON  = "square"

