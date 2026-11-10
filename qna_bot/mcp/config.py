"""MCP security defaults, allowlists, and runtime limits."""

from typing import FrozenSet

DEFAULT_CONNECT_TIMEOUT_SEC = 15
DEFAULT_TOOL_TIMEOUT_SEC = 60
MAX_TOOL_OUTPUT_CHARS = 16000
MAX_RESOURCE_BYTES = 512 * 1024

ALLOWED_STDIO_EXECUTABLES: FrozenSet[str] = frozenset(
    {
        "python",
        "python3",
        "uv",
        "uvx",
        "node",
        "npx",
        "docker",
    }
)

SENSITIVE_ARG_KEYS: FrozenSet[str] = frozenset(
    {
        "token",
        "access_token",
        "refresh_token",
        "api_key",
        "apikey",
        "secret",
        "password",
        "passwd",
        "authorization",
        "cookie",
        "credential",
        "private_key",
    }
)

BLOCKED_PARENT_ENV_KEYS: FrozenSet[str] = frozenset(
    {
        "GROQ_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "HF_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "GITHUB_TOKEN",
    }
)
