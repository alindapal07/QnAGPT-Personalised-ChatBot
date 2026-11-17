"""Authentication service providing Argon2id email/password login, Authlib OAuth2, and session verification."""

from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import os
from pathlib import Path
import secrets
from typing import Any, Callable, Dict, Optional

from authlib.integrations.httpx_client import OAuth2Client
from authlib.jose import JsonWebKey, jwt
from authlib.jose.errors import (
    BadSignatureError,
    DecodeError,
    ExpiredTokenError,
    InvalidClaimError,
    JoseError,
    UnsupportedAlgorithmError,
)
import httpx

from qna_bot.config.settings import AppSettings, get_settings
from qna_bot.core.exceptions import AuthenticationError, ConfigurationError, SecurityError
from qna_bot.core.logging_config import get_logger
from qna_bot.core.security import (
    SlidingWindowRateLimiter,
    generate_csrf_token,
    generate_session_token,
    get_dummy_password_hash,
    hash_password,
    hash_session_token,
    normalize_client_identifier,
    normalize_email,
    restrict_file_permissions,
    validate_password_strength,
    verify_password,
)
from qna_bot.database.legacy_migration import claim_legacy_workspace, has_unclaimed_legacy_data
from qna_bot.database.session import SQLAlchemyDatabase
from qna_bot.models.user import User
from qna_bot.repositories import (
    AuditAndTelemetryRepository,
    AuthRepository,
    UserRepository,
)
from qna_bot.schemas.auth import AuthResult, AuthenticatedSession, OAuthStartResult, UserProfile

logger = get_logger(__name__)

_DEFAULT_USER_SETTINGS: Dict[str, str] = {
    "text_model": "openai/gpt-oss-20b",
    "vision_model": "qwen/qwen3.8-27b",
    "temperature": "0.7",
    "max_tokens": "2048",
    "chat_mode": "Auto (Agentic)",
}

_OAUTH_PROVIDERS: Dict[str, Dict[str, str]] = {
    "google": {
        "authorize_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "userinfo_url": "https://openidconnect.googleapis.com/v1/userinfo",
        "scope": "openid email profile",
    },
    "github": {
        "authorize_url": "https://github.com/login/oauth/authorize",
        "token_url": "https://github.com/login/oauth/access_token",
        "userinfo_url": "https://api.github.com/user",
        "emails_url": "https://api.github.com/user/emails",
        "scope": "read:user user:email",
    },
}

_GOOGLE_JWKS_URL = "https://www.googleapis.com/oauth2/v3/certs"
_GOOGLE_ISSUERS = ["https://accounts.google.com", "accounts.google.com"]


def _to_user_profile(user: User) -> UserProfile:
    return UserProfile(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        avatar_url=user.avatar_url,
        is_active=user.is_active,
        is_verified=user.is_verified,
        has_password=bool(user.password_hash),
        created_at=user.created_at,
        last_login_at=user.last_login_at,
    )


def _pkce_challenge(verifier: str) -> str:
    import base64

    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


class AuthService:
    """Authoritative identity, OAuth2, and session manager backed by SQLite."""

    def __init__(
        self,
        db: Optional[SQLAlchemyDatabase] = None,
        settings: Optional[AppSettings] = None,
        rate_limiter: Optional[SlidingWindowRateLimiter] = None,
        oauth_client_factory: Optional[Callable[..., Any]] = None,
        google_jwks: Optional[Any] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.db = db or SQLAlchemyDatabase(self.settings.resolved_database_url)
        self.users = UserRepository(self.db)
        self.auth = AuthRepository(self.db)
        self.audit = AuditAndTelemetryRepository(self.db)
        max_attempts = int(getattr(self.settings, "auth_rate_limit_max_attempts", 5))
        window_sec = float(getattr(self.settings, "auth_rate_limit_window_sec", 300.0))
        self.rate_limiter = rate_limiter or SlidingWindowRateLimiter(
            max_attempts=max_attempts,
            window_sec=window_sec,
        )
        self._oauth_client_factory = oauth_client_factory or OAuth2Client
        self._google_jwks = google_jwks
        self._session_secret = self._resolve_session_secret()

    def _check_rate_limit(self, key: str, max_attempts: Optional[int] = None) -> None:
        """Safely invoke rate limiter check_allowed with backward-compatible argument handling."""
        if not self.rate_limiter:
            return
        try:
            if max_attempts is not None:
                self.rate_limiter.check_allowed(key, max_attempts=max_attempts)
            else:
                self.rate_limiter.check_allowed(key)
        except TypeError:
            # Fallback for rate limiter instances or mocks accepting only `key`
            self.rate_limiter.check_allowed(key)

    def _resolve_session_secret(self) -> str:
        explicit = (self.settings.session_secret_key or "").strip()
        if explicit:
            return explicit
        if ":memory:" in self.db.target:
            return secrets.token_urlsafe(32)
        key_path = Path(self.settings.data_dir) / ".session.key"
        if key_path.is_file():
            existing = key_path.read_text(encoding="utf-8").strip()
            if existing:
                restrict_file_permissions(key_path)
                return existing
        generated = secrets.token_urlsafe(48)
        try:
            key_path.parent.mkdir(parents=True, exist_ok=True)
            key_path.write_text(generated, encoding="utf-8")
            restrict_file_permissions(key_path)
        except OSError as exc:
            logger.warning("Could not persist .session.key file: %s", exc)
        return generated

    def _issue_session(self, user: User, is_new_user: bool = False) -> AuthResult:
        raw_token = generate_session_token()
        csrf_token = generate_csrf_token()
        token_hash = hash_session_token(raw_token, self._session_secret)
        expires_at = datetime.now(timezone.utc) + timedelta(hours=self.settings.session_ttl_hours)

        sess = self.auth.create_session(
            user_id=user.id,
            token_hash=token_hash,
            csrf_token=csrf_token,
            expires_at=expires_at,
        )
        self.users.mark_login(user.id)
        return AuthResult(
            session_token=raw_token,
            session=AuthenticatedSession(
                session_id=sess.id,
                user=_to_user_profile(user),
                csrf_token=csrf_token,
                expires_at=expires_at,
            ),
            is_new_user=is_new_user,
        )

    def _ensure_default_settings(self, user_id: str) -> None:
        existing = self.users.get_settings(user_id)
        if not existing:
            self.users.upsert_settings(
                user_id,
                {
                    **_DEFAULT_USER_SETTINGS,
                    "text_model": self.settings.default_text_model,
                    "vision_model": self.settings.default_vision_model,
                    "temperature": str(self.settings.default_temperature),
                    "max_tokens": str(self.settings.default_max_tokens),
                },
            )

    def register_with_password(
        self,
        email: str,
        password: str,
        full_name: str = "",
        client_key: str = "local",
    ) -> AuthResult:
        """Create a new local password account with Argon2id hashing under rate limits."""
        norm_client = normalize_client_identifier(client_key)
        reg_key = f"reg:{norm_client}"
        reg_max_attempts = int(getattr(self.settings, "auth_rate_limit_registration_max_attempts", 10))
        self._check_rate_limit(
            reg_key,
            max_attempts=reg_max_attempts,
        )

        norm_email = normalize_email(email)
        validate_password_strength(password)

        existing_user = self.users.get_by_email(norm_email)
        if existing_user is not None:
            self.rate_limiter.record_failure(reg_key)
            self.audit.log_audit(
                user_id=existing_user.id,
                action="auth.register_duplicate_attempt",
                resource_type="user",
                resource_id=norm_email,
                metadata={"method": "password", "client": norm_client},
            )
            raise AuthenticationError(
                "Registration could not be completed with this email address. If you already have an account, please sign in."
            )

        pw_hash = hash_password(password)
        user = self.users.create_user(
            email=norm_email,
            password_hash=pw_hash,
            full_name=(full_name or norm_email.split("@")[0]).strip()[:160],
            is_verified=False,
        )
        self.rate_limiter.record_failure(reg_key)
        self._ensure_default_settings(user.id)
        self.audit.log_audit(
            user_id=user.id,
            action="auth.register",
            resource_type="user",
            resource_id=user.id,
            metadata={"method": "password", "client": norm_client},
        )
        return self._issue_session(user, is_new_user=True)

    def login_with_password(
        self,
        email: str,
        password: str,
        client_key: str = "local",
    ) -> AuthResult:
        """Verify email + Argon2id password under multi-dimensional rate limits.

        Enforces separate thresholds for:
        1. Client-Account pair limits: caps attempts by this client on this account (e.g. 5 failures / window)
        2. Client-wide limits: caps total failures across all accounts from this client (e.g. 20 failures / window)
        3. Fair account protection: tracks global account failure volume without locking out the legitimate owner
           who supplies the correct password.

        Performs constant-work password verification for all authentication requests
        (including nonexistent users, inactive accounts, and passwordless accounts)
        using a valid dummy Argon2id hash to mitigate password timing attacks and
        account enumeration.
        """
        raw_email = (email or "").strip().lower()
        norm_client = normalize_client_identifier(client_key)
        client_tag = f"client:{norm_client}"
        pair_key = f"pair:{norm_client}:{raw_email or 'empty'}"
        legacy_key = f"pw:{raw_email or 'empty'}:{client_key}"

        # Dimension A: Check client-wide failure threshold (blocks credential stuffing across accounts)
        client_max_attempts = int(getattr(self.settings, "auth_rate_limit_client_max_attempts", 20))
        self._check_rate_limit(
            client_tag,
            max_attempts=client_max_attempts,
        )

        # Dimension B: Check client-account pair threshold (blocks targeted guessing from this client)
        pair_max_attempts = int(getattr(self.settings, "auth_rate_limit_max_attempts", 5))
        self._check_rate_limit(
            pair_key,
            max_attempts=pair_max_attempts,
        )

        # Dimension C: Check legacy rate key for backwards compatibility
        self._check_rate_limit(legacy_key)

        dummy_hash = get_dummy_password_hash()

        try:
            norm_email = normalize_email(raw_email)
        except AuthenticationError:
            verify_password(password or "dummy", dummy_hash)
            self.rate_limiter.record_failure(client_tag)
            self.rate_limiter.record_failure(pair_key)
            self.rate_limiter.record_failure(legacy_key)
            raise AuthenticationError("Invalid email or password.")

        user = self.users.get_by_email(norm_email)

        # Only an existing active account with an established password hash can be authenticated
        is_real_candidate = (
            user is not None
            and user.is_active
            and bool(user.password_hash)
        )

        if is_real_candidate:
            is_valid_pw = verify_password(password, user.password_hash)
        else:
            # Execute dummy verification to equalize computational workload across invalid categories
            verify_password(password or "dummy", dummy_hash)
            is_valid_pw = False

        if not is_real_candidate or not is_valid_pw:
            self.rate_limiter.record_failure(client_tag)
            self.rate_limiter.record_failure(pair_key)
            self.rate_limiter.record_failure(legacy_key)
            acct_fails = self.rate_limiter.record_failure(f"acct:{norm_email}")
            if isinstance(acct_fails, (int, float)) and acct_fails >= 10:
                logger.warning(
                    "High authentication failure volume on account %s (%d failures in active window).",
                    norm_email,
                    acct_fails,
                )
            self.audit.log_audit(
                user_id=user.id if user else None,
                action="auth.login_failed",
                resource_type="user",
                resource_id=norm_email,
                metadata={"client": norm_client},
            )
            raise AuthenticationError("Invalid email or password.")

        # Successful authentication: clear client-account failure counters
        self.rate_limiter.clear(pair_key)
        self.rate_limiter.clear(legacy_key)
        self._ensure_default_settings(user.id)
        self.audit.log_audit(
            user_id=user.id,
            action="auth.login",
            resource_type="user",
            resource_id=user.id,
            metadata={"method": "password"},
        )
        return self._issue_session(user, is_new_user=False)

    def validate_session(self, session_token: Optional[str]) -> Optional[AuthenticatedSession]:
        """Resolve an opaque session token strictly from backend SQLite state."""
        if not session_token:
            return None
        token_hash = hash_session_token(session_token, self._session_secret)
        row = self.auth.get_active_session_with_user(token_hash)
        if row is None:
            return None
        sess, user = row
        return AuthenticatedSession(
            session_id=sess.id,
            user=_to_user_profile(user),
            csrf_token=sess.csrf_token,
            expires_at=sess.expires_at,
        )

    def require_session(self, session_token: Optional[str]) -> AuthenticatedSession:
        """Require a valid authenticated session or raise AuthenticationError."""
        resolved = self.validate_session(session_token)
        if resolved is None:
            raise AuthenticationError("Authentication required. Please sign in to continue.")
        return resolved

    def logout(self, session_token: Optional[str]) -> bool:
        """Revoke the current persistent session token immediately."""
        if not session_token:
            return False
        token_hash = hash_session_token(session_token, self._session_secret)
        existing = self.auth.get_active_session_with_user(token_hash)
        revoked = self.auth.revoke_session_by_hash(token_hash)
        if revoked and existing is not None:
            self.audit.log_audit(
                user_id=existing[1].id,
                action="auth.logout",
                resource_type="user_session",
                resource_id=existing[0].id,
            )
        return revoked

    def _provider_credentials(self, provider: str) -> tuple[str, str]:
        prov = (provider or "").strip().lower()
        if prov == "google":
            cid = self.settings.google_client_id.strip()
            sec = self.settings.google_client_secret.strip()
            if not cid or not sec:
                raise ConfigurationError(
                    "Google Sign-In is not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in your .env file."
                )
            return cid, sec
        if prov == "github":
            cid = self.settings.github_client_id.strip()
            sec = self.settings.github_client_secret.strip()
            if not cid or not sec:
                raise ConfigurationError(
                    "GitHub Sign-In is not configured. Set GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET in your .env file."
                )
            return cid, sec
        raise ConfigurationError(f"Unsupported OAuth provider '{provider}'.")

    def is_oauth_configured(self, provider: str) -> bool:
        try:
            self._provider_credentials(provider)
            return True
        except ConfigurationError:
            return False

    def resolve_provider_from_state(self, state: str) -> Optional[str]:
        """Resolve the registered OAuth provider for an active pending CSRF state token."""
        if not state:
            return None
        pending = self.auth.get_pending_state(state)
        return pending.provider if pending else None

    def oauth_redirect_uri(self, provider: str) -> str:
        prov = (provider or "").strip().lower()
        if prov == "google" and getattr(self.settings, "google_redirect_uri", ""):
            return self.settings.google_redirect_uri.strip()
        if prov == "github" and getattr(self.settings, "github_redirect_uri", ""):
            return self.settings.github_redirect_uri.strip()

        base = (self.settings.app_base_url or "http://localhost:8501").rstrip("/")
        # Google OAuth strictly forbids query parameters in Authorized redirect URIs in Google Cloud Console.
        # Google OAuth callbacks must route to the root URL (e.g. http://localhost:8501/).
        if prov == "google":
            return f"{base}/"
        return f"{base}/?oauth_provider={prov}"

    def start_oauth_login(self, provider: str) -> OAuthStartResult:
        """Build a real Authlib OAuth2/PKCE authorization URL and persist single-use CSRF state."""
        prov = (provider or "").strip().lower()
        meta = _OAUTH_PROVIDERS.get(prov)
        if meta is None:
            raise ConfigurationError(f"Unsupported OAuth provider '{provider}'.")
        client_id, client_secret = self._provider_credentials(prov)
        redirect_uri = self.oauth_redirect_uri(prov)
        state = secrets.token_urlsafe(32)
        code_verifier = secrets.token_urlsafe(48)
        nonce = secrets.token_urlsafe(24)

        client = self._oauth_client_factory(
            client_id=client_id,
            client_secret=client_secret,
            scope=meta["scope"],
            redirect_uri=redirect_uri,
        )
        url_kwargs: Dict[str, Any] = {
            "state": state,
            "code_challenge": _pkce_challenge(code_verifier),
            "code_challenge_method": "S256",
        }
        if prov == "google":
            url_kwargs["nonce"] = nonce
            url_kwargs["access_type"] = "offline"
            url_kwargs["prompt"] = "select_account"

        authorization_url, _ = client.create_authorization_url(meta["authorize_url"], **url_kwargs)
        self.auth.create_oauth_state(
            state=state,
            provider=prov,
            code_verifier=code_verifier,
            nonce=nonce,
            redirect_uri=redirect_uri,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
        )
        return OAuthStartResult(
            provider=prov,
            authorization_url=authorization_url,
            state=state,
            redirect_uri=redirect_uri,
        )

    def complete_oauth_login(
        self,
        provider: str,
        code: str,
        state: str,
    ) -> AuthResult:
        """Validate single-use CSRF state, exchange code server-side via Authlib, and create user session."""
        prov = (provider or "").strip().lower()
        if not prov and state:
            resolved = self.resolve_provider_from_state(state)
            if resolved:
                prov = resolved
        meta = _OAUTH_PROVIDERS.get(prov)
        if meta is None:
            raise ConfigurationError(f"Unsupported OAuth provider '{provider}'.")
        client_id, client_secret = self._provider_credentials(prov)
        if not code or not state:
            raise AuthenticationError("OAuth callback is missing required authorization code or state.")

        pending = self.auth.consume_oauth_state(prov, state)
        if pending is None:
            raise SecurityError("Invalid, expired, or already-used OAuth CSRF state parameter.")

        client = self._oauth_client_factory(
            client_id=client_id,
            client_secret=client_secret,
            scope=meta["scope"],
            redirect_uri=pending.redirect_uri,
        )
        try:
            token_kwargs: Dict[str, Any] = {
                "url": meta["token_url"],
                "code": code,
                "code_verifier": pending.code_verifier,
            }
            if prov == "github":
                token = client.fetch_token(**token_kwargs, headers={"Accept": "application/json"})
            else:
                token = client.fetch_token(**token_kwargs)
        except Exception as exc:
            logger.warning("OAuth token exchange failed for provider=%s: %s", prov, exc)
            raise AuthenticationError(f"Failed to complete {prov.title()} authentication: {exc}") from exc

        if not isinstance(token, dict):
            raise AuthenticationError(f"{prov.title()} token response was invalid.")

        if prov == "google":
            raw_id_token = token.get("id_token")
            if not raw_id_token:
                raise AuthenticationError("Google OAuth code exchange did not return an ID token.")

            claims = self._validate_google_id_token(
                raw_id_token=raw_id_token,
                expected_nonce=pending.nonce,
                client_id=client_id,
            )
            sub = str(claims.get("sub") or "").strip()
            email = normalize_email(str(claims.get("email") or ""))
            email_verified = bool(claims.get("email_verified", False))
            full_name = str(claims.get("name") or email.split("@")[0])[:160]
            avatar_url = str(claims.get("picture") or "")[:512]
        else:
            profile = self._fetch_oauth_userinfo(prov, client, meta)
            sub = str(profile["sub"])
            email = normalize_email(profile["email"])
            email_verified = bool(profile["email_verified"])
            full_name = str(profile.get("name") or email.split("@")[0])[:160]
            avatar_url = str(profile.get("avatar_url") or "")[:512]

        identity = self.auth.get_identity(prov, sub)
        is_new_user = False
        if identity is not None:
            user = self.users.get_by_id(identity.user_id)
            if user is None or not user.is_active:
                raise AuthenticationError("Linked account is inactive or no longer exists.")
        else:
            existing_user = self.users.get_by_email(email)
            if existing_user is not None:
                if not email_verified:
                    raise AuthenticationError(
                        f"Cannot link {prov.title()} account automatically because your provider email is unverified."
                    )
                user = existing_user
                if avatar_url and not user.avatar_url:
                    user = self.users.update_profile(user.id, avatar_url=avatar_url)
            else:
                user = self.users.create_user(
                    email=email,
                    password_hash=None,
                    full_name=full_name,
                    avatar_url=avatar_url,
                    is_verified=email_verified,
                )
                is_new_user = True
            self.auth.link_identity(
                user_id=user.id,
                provider=prov,
                provider_user_id=sub,
                provider_email=email,
                email_verified=email_verified,
            )

        self.users.mark_login(user.id, mark_verified=email_verified)
        self._ensure_default_settings(user.id)
        self.audit.log_audit(
            user_id=user.id,
            action="auth.oauth_login",
            resource_type="auth_identity",
            resource_id=prov,
            metadata={"provider": prov, "email_verified": email_verified},
        )
        return self._issue_session(user, is_new_user=is_new_user)

    def _get_google_jwks(self) -> Any:
        if self._google_jwks is not None:
            if isinstance(self._google_jwks, dict):
                return JsonWebKey.import_key_set(self._google_jwks)
            return self._google_jwks

        try:
            resp = httpx.get(_GOOGLE_JWKS_URL, timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
            return JsonWebKey.import_key_set(data)
        except Exception as exc:
            logger.warning("Failed fetching Google JWKS from %s: %s", _GOOGLE_JWKS_URL, exc)
            raise AuthenticationError("Unable to retrieve Google public verification keys.") from exc

    def _validate_google_id_token(
        self,
        raw_id_token: str,
        expected_nonce: str,
        client_id: str,
    ) -> Dict[str, Any]:
        """Cryptographically validate Google OpenID Connect ID token and enforce nonce match."""
        if not raw_id_token or not isinstance(raw_id_token, str):
            raise AuthenticationError("Google authentication failed: missing ID token in response.")

        keys = self._get_google_jwks()

        claims_options = {
            "iss": {
                "essential": True,
                "values": _GOOGLE_ISSUERS,
            },
            "aud": {
                "essential": True,
                "value": client_id,
            },
        }

        try:
            claims = jwt.decode(raw_id_token, keys, claims_options=claims_options)
            claims.validate(leeway=120)
        except ExpiredTokenError as exc:
            logger.warning("Google ID token expired: %s", exc)
            raise AuthenticationError("Google ID token has expired. Please sign in again.") from exc
        except InvalidClaimError as exc:
            logger.warning("Google ID token invalid claim: %s", exc)
            raise AuthenticationError(f"Google ID token claim validation failed: {exc}") from exc
        except (BadSignatureError, UnsupportedAlgorithmError, DecodeError) as exc:
            logger.warning("Google ID token signature verification failed: %s", exc)
            raise AuthenticationError("Google ID token signature verification failed.") from exc
        except (JoseError, ValueError) as exc:
            logger.warning("Google ID token JOSE decoding or key matching failed: %s", exc)
            raise AuthenticationError("Google ID token could not be verified.") from exc

        # Strict Nonce Validation
        token_nonce = claims.get("nonce")
        if not token_nonce:
            logger.warning("Google ID token missing required nonce claim.")
            raise AuthenticationError("Google ID token is missing required nonce claim.")

        if not expected_nonce:
            logger.warning("Pending OAuth state is missing expected nonce.")
            raise SecurityError("OAuth state is missing expected nonce.")

        if not hmac.compare_digest(str(token_nonce), str(expected_nonce)):
            logger.warning("Google OAuth nonce mismatch.")
            raise SecurityError("Google OAuth nonce mismatch.")

        return dict(claims)

    def _fetch_oauth_userinfo(
        self,
        provider: str,
        client: Any,
        meta: Dict[str, str],
    ) -> Dict[str, Any]:
        resp = client.get(meta["userinfo_url"])
        resp.raise_for_status()
        data = resp.json()
        if provider == "google":
            sub = str(data.get("sub") or "").strip()
            email = str(data.get("email") or "").strip().lower()
            verified = bool(data.get("email_verified", False))
            if not sub or not email:
                raise AuthenticationError("Google userinfo did not return account identity and email.")
            return {
                "sub": sub,
                "email": email,
                "email_verified": verified,
                "name": str(data.get("name") or ""),
                "avatar_url": str(data.get("picture") or ""),
            }

        sub = str(data.get("id") or "").strip()
        email = str(data.get("email") or "").strip().lower()
        verified = False
        if "emails_url" in meta:
            try:
                eresp = client.get(meta["emails_url"])
                if eresp.status_code == 200:
                    emails = eresp.json() or []
                    primary_verified = next(
                        (e for e in emails if isinstance(e, dict) and e.get("primary") and e.get("verified")),
                        None,
                    )
                    any_verified = next(
                        (e for e in emails if isinstance(e, dict) and e.get("verified")),
                        None,
                    )
                    chosen = primary_verified or any_verified
                    if chosen:
                        email = str(chosen.get("email") or email).strip().lower()
                        verified = bool(chosen.get("verified", False))
            except Exception as exc:
                logger.warning("GitHub emails endpoint failed: %s", exc)
        if not sub or not email:
            raise AuthenticationError(
                "GitHub profile did not supply a readable primary email address."
            )
        return {
            "sub": sub,
            "email": email,
            "email_verified": verified,
            "name": str(data.get("name") or data.get("login") or ""),
            "avatar_url": str(data.get("avatar_url") or ""),
        }

    def has_unclaimed_legacy_data(self) -> bool:
        return has_unclaimed_legacy_data(self.db)

    def claim_legacy_workspace_for_session(self, session_token: str) -> Dict[str, int]:
        """Claim legacy local_user data exclusively for the verified active session user."""
        session = self.require_session(session_token)
        moved = claim_legacy_workspace(self.db, session.user.id)
        self.audit.log_audit(
            user_id=session.user.id,
            action="workspace.claim_legacy",
            resource_type="user",
            resource_id=session.user.id,
            metadata=moved,
        )
        return moved
