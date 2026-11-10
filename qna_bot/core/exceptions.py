"""Domain-specific exceptions for the QnA Chatbot application."""


class QnABotError(Exception):
    """Base exception for all QnA Chatbot errors."""


class ConfigurationError(QnABotError):
    """Raised when required configuration or credentials are missing or invalid."""


class ImageValidationError(QnABotError):
    """Raised when an uploaded image fails size, format, or integrity validation."""


class AgentExecutionError(QnABotError):
    """Raised when the AI agent encounters an error while generating a response."""


class DocumentIngestionError(QnABotError):
    """Raised when document validation, loading, or chunking fails."""


class EmbeddingError(QnABotError):
    """Raised when the local embedding model fails to initialize or encode text."""


class VectorStoreError(QnABotError):
    """Raised when vector index creation, loading, persistence, or search fails."""


class ToolExecutionError(QnABotError):
    """Raised when an external or retrieval tool encounters an unrecoverable error."""


class MemoryPersistenceError(QnABotError):
    """Raised when SQLite checkpointing or thread persistence fails."""


class AuthenticationError(QnABotError):
    """Raised when user authentication or session validation fails."""


class AuthorizationError(QnABotError):
    """Raised when an authenticated user attempts an unauthorized resource action."""


class SecurityError(QnABotError):
    """Raised when a security boundary (path traversal, rate limit, SSRF) is violated."""


class StorageError(QnABotError):
    """Raised when local storage persistence or retrieval fails."""


class DatabaseError(QnABotError):
    """Raised when relational database transactions or migrations fail."""
