"""Cryptographic password hashing, email/password validation, signed session tokens, and rate limiting."""

from collections import defaultdict, deque
import hashlib
import hmac
import ipaddress
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import subprocess
import sys
import threading
import time
from typing import Deque, Dict, Optional, Union

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

from qna_bot.core.exceptions import AuthenticationError, SecurityError
from qna_bot.core.logging_config import get_logger

logger = get_logger(__name__)

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


_DUMMY_PASSWORD_HASH: Optional[str] = None
_DUMMY_LOCK = threading.Lock()


def get_dummy_password_hash() -> str:
    """Return a valid Argon2id hash generated with the application's configured hasher.

    Lazily initialized on first use with a cryptographically secure random token,
    ensuring exact parameter compatibility with _PASSWORD_HASHER while avoiding
    unnecessary overhead at import time.
    """
    global _DUMMY_PASSWORD_HASH
    if _DUMMY_PASSWORD_HASH is None:
        with _DUMMY_LOCK:
            if _DUMMY_PASSWORD_HASH is None:
                _DUMMY_PASSWORD_HASH = _PASSWORD_HASHER.hash(secrets.token_urlsafe(32))
    return _DUMMY_PASSWORD_HASH


def generate_session_token() -> str:
    """Generate a high-entropy opaque session bearer token."""
    return secrets.token_urlsafe(32)


def generate_csrf_token() -> str:
    """Generate a CSRF token bound to a user session."""
    return secrets.token_urlsafe(24)


def hash_session_token(raw_token: str, secret_key: str) -> str:
    """Compute keyed HMAC-SHA256 digest of a raw session token for SQLite storage.
    
    We never store raw session bearer tokens in the database. If an attacker gains
    read access to the SQLite file, they cannot hijack active sessions without the
    secret key used to compute this HMAC digest.
    """
    if not raw_token:
        return ""
    if not secret_key or not str(secret_key).strip():
        raise SecurityError(
            "A non-empty secret key is required for session token hashing. "
            "Configure SESSION_SECRET_KEY in the environment or ensure the active "
            "AuthService session secret is initialized."
        )
    key_bytes = str(secret_key).encode("utf-8")
    return hmac.new(key_bytes, raw_token.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hmac_signature(value: str, expected_mac: str, secret_key: str) -> bool:
    """Verify an HMAC digest using constant-time comparison to prevent timing attacks."""
    if not secret_key or not str(secret_key).strip():
        raise SecurityError("A non-empty secret key is required to verify HMAC signatures.")
    if not value or not expected_mac:
        return False
    actual = hash_session_token(value, secret_key)
    return hmac.compare_digest(actual, expected_mac)


def restrict_file_permissions(path: Union[str, Path]) -> bool:
    """Restrict filesystem permissions on a sensitive key, secret, or database file.

    Cross-platform behavior:
    - On POSIX (Linux/macOS): Applies mode 0600 (owner read/write only) for files,
      or 0700 for directories.
    - On Windows (NTFS): Uses the Windows-native `icacls` utility to remove inherited
      permissions (/inheritance:r) and grant Full control strictly to the current user
      account (/grant:r %USERNAME%:F), preventing access by other local users on shared machines.
    """
    target = Path(path).resolve()
    if not target.exists():
        logger.warning("Cannot restrict permissions on non-existent path: %s", target)
        return False

    if sys.platform != "win32":
        # POSIX (Linux, macOS, BSD, Docker containers)
        try:
            mode = 0o700 if target.is_dir() else 0o600
            os.chmod(target, mode)
            return True
        except OSError as exc:
            logger.warning("Could not set POSIX permissions on %s: %s", target, exc)
            return False

    # Windows (NTFS Access Control Lists)
    success = True
    try:
        os.chmod(target, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        success = False

    icacls_bin = shutil.which("icacls") or os.path.join(
        os.environ.get("SystemRoot", "C:\\Windows"), "System32", "icacls.exe"
    )
    if os.path.exists(icacls_bin):
        import getpass

        user = os.environ.get("USERNAME") or getpass.getuser()
        try:
            cmd = [icacls_bin, str(target), "/inheritance:r", "/grant:r", f"{user}:F"]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                logger.warning(
                    "icacls failed to restrict Windows ACL on %s: %s",
                    target.name,
                    proc.stderr.strip() or proc.stdout.strip(),
                )
                success = False
        except Exception as exc:
            logger.warning("Exception executing icacls on %s: %s", target.name, exc)
            success = False
    else:
        logger.warning("icacls.exe not found on Windows; ACL restriction skipped for %s", target.name)
        success = False

    return success


def normalize_client_identifier(
    client_raw: Optional[str] = None,
    trusted_proxies: Optional[Union[str, list[str]]] = None,
    headers: Optional[Dict[str, str]] = None,
    session_id: Optional[str] = None,
) -> str:
    """Safely normalize and anonymize client identifiers for rate limiting.

    - Canonicalizes valid IPv4 and IPv6 addresses.
    - Only trusts X-Forwarded-For when client_raw explicitly matches a configured trusted proxy.
    - Incorporates persistent session tokens when direct network IP is unavailable.
    - Derives a deterministic truncated SHA-256 token to prevent storing raw PII in limiter state.
    """
    raw = (client_raw or "").strip()
    candidate_ip: Optional[str] = None

    if raw:
        try:
            parsed = ipaddress.ip_address(raw)
            candidate_ip = str(parsed)
        except ValueError:
            pass

    # Check trusted proxy boundary if candidate_ip is a configured proxy
    if candidate_ip and trusted_proxies and headers:
        trusted_set = set()
        if isinstance(trusted_proxies, str):
            trusted_set = {p.strip() for p in trusted_proxies.split(",") if p.strip()}
        elif isinstance(trusted_proxies, (list, tuple, set)):
            trusted_set = {str(p).strip() for p in trusted_proxies if str(p).strip()}

        if candidate_ip in trusted_set:
            xff = headers.get("x-forwarded-for") or headers.get("X-Forwarded-For") or ""
            if xff:
                # Traverse hops right-to-left to find the first untrusted upstream client
                hops = [h.strip() for h in xff.split(",") if h.strip()]
                for hop in reversed(hops):
                    try:
                        hop_ip = ipaddress.ip_address(hop)
                        if str(hop_ip) not in trusted_set:
                            candidate_ip = str(hop_ip)
                            break
                    except ValueError:
                        continue

    if candidate_ip:
        token = f"ip:{candidate_ip}"
    elif session_id and session_id.strip():
        token = f"sess:{session_id.strip()}"
    elif raw:
        sanitized = re.sub(r"[^a-zA-Z0-9_\-\.:]", "", raw)[:64]
        token = f"id:{sanitized}" if sanitized else "unknown"
    else:
        token = "unknown"

    # Deterministic privacy-preserving client key
    client_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
    return f"c_{client_hash}"


class SlidingWindowRateLimiter:
    """Thread-safe in-memory rate limiter for authentication attempts with memory bounds.
    
    Uses a sliding time window tracked via collections.deque with monotonic timestamps.
    time.monotonic() is intentionally used instead of time.time() to ensure clock
    adjustments (e.g., NTP synchronization) cannot bypass or prematurely lock out accounts.
    """

    def __init__(
        self,
        max_attempts: int = 5,
        window_sec: float = 300.0,
        max_keys: int = 10000,
    ) -> None:
        self.max_attempts = max(1, int(max_attempts))
        self.window_sec = max(0.05, float(window_sec))
        self.max_keys = max(10, int(max_keys))
        self._lock = threading.Lock()
        self._attempts: Dict[str, Deque[float]] = defaultdict(deque)

    def _prune_expired_under_lock(self, now: float) -> None:
        """Prune empty and fully expired keys to prevent unbounded memory growth."""
        expired_keys = []
        for k, dq in self._attempts.items():
            while dq and (now - dq[0]) > self.window_sec:
                dq.popleft()
            if not dq:
                expired_keys.append(k)
        for k in expired_keys:
            self._attempts.pop(k, None)

    def check_allowed(self, key: str, max_attempts: Optional[int] = None) -> None:
        """Raise SecurityError if key has exceeded the allowed failure threshold."""
        threshold = max_attempts if max_attempts is not None else self.max_attempts
        now = time.monotonic()
        with self._lock:
            dq = self._attempts.get(key)
            if not dq:
                return
            # Prune attempts that have fallen outside the rolling window
            while dq and (now - dq[0]) > self.window_sec:
                dq.popleft()
            if not dq:
                self._attempts.pop(key, None)
                return
            if len(dq) >= threshold:
                retry_in = int(max(1, self.window_sec - (now - dq[0])))
                raise SecurityError(
                    f"Too many failed authentication attempts. Please try again in {retry_in}s."
                )

    def record_failure(self, key: str) -> int:
        """Record an authentication failure timestamp under lock and return current count in window."""
        now = time.monotonic()
        with self._lock:
            # Opportunistic cleanup if key count exceeds capacity bounds
            if len(self._attempts) >= self.max_keys:
                self._prune_expired_under_lock(now)

            dq = self._attempts[key]
            while dq and (now - dq[0]) > self.window_sec:
                dq.popleft()
            dq.append(now)
            return len(dq)

    def get_failure_count(self, key: str) -> int:
        """Get the active count of failures within the rolling window without raising."""
        now = time.monotonic()
        with self._lock:
            dq = self._attempts.get(key)
            if not dq:
                return 0
            while dq and (now - dq[0]) > self.window_sec:
                dq.popleft()
            if not dq:
                self._attempts.pop(key, None)
                return 0
            return len(dq)

    def clear(self, key: str) -> None:
        """Reset failed attempts for key upon successful authentication."""
        with self._lock:
            self._attempts.pop(key, None)
