"""Centralized logging configuration for the QnA Chatbot."""

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_ROOT_LOGGER_NAME = "qna_bot"


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the application root logger idempotently."""
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    resolved_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(resolved_level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the application root namespace."""
    if name.startswith(f"{_ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
