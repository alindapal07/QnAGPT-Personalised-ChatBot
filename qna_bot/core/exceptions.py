"""Domain-specific exceptions for the QnA Chatbot application."""


class QnABotError(Exception):
    """Base exception for all QnA Chatbot errors."""


class ConfigurationError(QnABotError):
    """Raised when required configuration or credentials are missing or invalid."""


class ImageValidationError(QnABotError):
    """Raised when an uploaded image fails size, format, or integrity validation."""


class AgentExecutionError(QnABotError):
    """Raised when the AI agent encounters an error while generating a response."""
