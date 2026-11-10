"""Cryptographic password hashing, email/password validation, signed session tokens, and rate limiting."""

from collections import defaultdict, deque
import hashlib
import hmac
import re
import secrets
import threading
import time
from typing import Deque, Dict

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

from qna_bot.core.exceptions import AuthenticationError, SecurityError

_PASSWORD_HASHER = PasswordHasher()
_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")


def normalize_email(email: str) -> str:
    """Normalize and validate an email address format."""
    cleaned = (email or "").strip().lower()
    if not cleaned or len(cleaned) > 254 or not _EMAIL_RE.match(cleaned):
        raise AuthenticationError("Please provide a valid email address.")
    return cleaned


def validate_password_strength(password: str) -> str:
    """Enforce baseline password complexity without corrupting valid characters."""
    if not isinstance(password, str) or len(password) < 8:
        raise AuthenticationError("Password must be at least 8 characters long.")
    if len(password) > 128:
        raise AuthenticationError("Password must not exceed 128 characters.")
    has_upper = any(ch.isupper() for ch in password)
    has_lower = any(ch.islower() for ch in password)
    has_digit_or_sym = any(ch.isdigit() or not ch.isalnum() for ch in password)
    if not (has_upper and has_lower and has_digit_or_sym):
        raise AuthenticationError(
            "Password must include uppercase letters, lowercase letters, and a number or symbol."
        )
    return password


def hash_password(password: str) -> str:
    """Hash a plaintext password using Argon2id with a cryptographic random salt."""
    return _PASSWORD_HASHER.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Constant-time Argon2id password verification."""
    if not password or not password_hash:
        return False
    try:
        return _PASSWORD_HASHER.verify(password_hash, password)
    except VerifyMismatchError:
        return False
    except Exception:
        return False


def generate_session_token() -> str:
    """Generate a high-entropy opaque session bearer token."""
    return secrets.token_urlsafe(32)


def generate_csrf_token() -> str:
    """Generate a CSRF token bound to a user session."""
    return secrets.token_urlsafe(24)


def hash_session_token(raw_token: str, secret_key: str) -> str:
    """Compute keyed HMAC-SHA256 digest of a raw session token for SQLite storage."""
    if not raw_token:
        return ""
    key_bytes = (secret_key or "qnagpt-local-session-key").encode("utf-8")
    return hmac.new(key_bytes, raw_token.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hmac_signature(value: str, expected_mac: str, secret_key: str) -> bool:
    """Verify an HMAC digest using constant-time comparison."""
    actual = hash_session_token(value, secret_key)
    return hmac.compare_digest(actual, expected_mac or "")


class SlidingWindowRateLimiter:
    """Thread-safe in-memory rate limiter for authentication attempts."""

    def __init__(self, max_attempts: int = 5, window_sec: float = 300.0) -> None:
        self.max_attempts = max(1, int(max_attempts))
        self.window_sec = max(1.0, float(window_sec))
        self._lock = threading.Lock()
        self._attempts: Dict[str, Deque[float]] = defaultdict(deque)

    def check_allowed(self, key: str) -> None:
        """Raise SecurityError if key has exceeded the allowed failure threshold."""
        now = time.monotonic()
        with self._lock:
            dq = self._attempts[key]
            while dq and (now - dq[0]) > self.window_sec:
                dq.popleft()
            if len(dq) >= self.max_attempts:
                retry_in = int(max(1, self.window_sec - (now - dq[0])))
                raise SecurityError(
                    f"Too many failed authentication attempts. Please try again in {retry_in}s."
                )

    def record_failure(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            dq = self._attempts[key]
            while dq and (now - dq[0]) > self.window_sec:
                dq.popleft()
            dq.append(now)

    def clear(self, key: str) -> None:
        with self._lock:
            self._attempts.pop(key, None)
