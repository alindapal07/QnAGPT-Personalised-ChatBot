"""Encrypted MCP credential vault (Fernet AES-128-CBC + HMAC-SHA256) and SDK TokenStorage."""

import base64
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional
import uuid

from cryptography.fernet import Fernet, InvalidToken
from mcp.client.auth.oauth2 import TokenStorage
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from qna_bot.config.settings import AppSettings, get_settings
from qna_bot.core.logging_config import get_logger
from qna_bot.database.connection import DatabaseManager
from qna_bot.database.models import EncryptedCredentialRecord
from qna_bot.database.repositories import CredentialRepository
from qna_bot.mcp.exceptions import MCPSecurityError

logger = get_logger(__name__)


def _derive_fernet_key(raw_key: str) -> bytes:
    """Normalize an env-supplied Fernet key or passphrase into a valid 32-byte URL-safe base64 key."""
    candidate = (raw_key or "").strip().encode("utf-8")
    try:
        Fernet(candidate)
        return candidate
    except Exception:
        digest = hashlib.sha256(candidate).digest()
        return base64.urlsafe_b64encode(digest)


class CredentialVault:
    """Encrypts and persists MCP secrets and OAuth tokens strictly per `(user_id, server_id)`."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        encryption_key: Optional[str] = None,
        settings: Optional[AppSettings] = None,
    ) -> None:
        self.db = db_manager or DatabaseManager()
        self.repo = CredentialRepository(self.db)
        self._settings = settings or get_settings()
        self._key = self._resolve_master_key(encryption_key)
        self._fernet = Fernet(self._key)
        self.key_fingerprint = hashlib.sha256(self._key).hexdigest()[:16]

    def _resolve_master_key(self, explicit_key: Optional[str]) -> bytes:
        raw = (explicit_key or self._settings.credential_encryption_key or "").strip()
        if raw:
            return _derive_fernet_key(raw)

        db_path_str = str(self.db.db_path)
        if db_path_str == ":memory:":
            return Fernet.generate_key()

        key_path = Path(self.db.db_path).parent / ".mcp_vault.key"
        if key_path.is_file():
            existing = key_path.read_text(encoding="utf-8").strip()
            if existing:
                return _derive_fernet_key(existing)

        generated = Fernet.generate_key()
        try:
            key_path.parent.mkdir(parents=True, exist_ok=True)
            key_path.write_text(generated.decode("utf-8"), encoding="utf-8")
            try:
                os.chmod(key_path, 0o600)
            except OSError:
                pass
        except OSError as exc:
            logger.warning("Could not persist MCP vault key file: %s", exc)
        return generated

    def encrypt_dict(self, payload: Mapping[str, Any]) -> str:
        raw_json = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return self._fernet.encrypt(raw_json).decode("utf-8")

    def decrypt_dict(self, ciphertext: str) -> Dict[str, Any]:
        if not ciphertext:
            return {}
        try:
            plaintext = self._fernet.decrypt(ciphertext.encode("utf-8"))
            parsed = json.loads(plaintext.decode("utf-8"))
            return parsed if isinstance(parsed, dict) else {}
        except (InvalidToken, ValueError) as exc:
            raise MCPSecurityError(
                "Stored MCP credentials could not be decrypted with the active vault key."
            ) from exc

    def store_secret(
        self,
        user_id: str,
        server_id: str,
        auth_type: str,
        secret_payload: Mapping[str, Any],
        header_name: str = "Authorization",
        expires_at: Optional[float] = None,
        scopes: str = "",
    ) -> EncryptedCredentialRecord:
        """Encrypt and persist credentials at rest. Never stores plaintext tokens."""
        ciphertext = self.encrypt_dict(secret_payload)
        record = EncryptedCredentialRecord(
            credential_id=f"cred_{uuid.uuid4().hex[:12]}",
            user_id=user_id,
            server_id=server_id,
            auth_type=auth_type,
            header_name=header_name or "Authorization",
            encrypted_payload=ciphertext,
            key_fingerprint=self.key_fingerprint,
            expires_at=expires_at,
            scopes=scopes,
        )
        return self.repo.upsert(record)

    def load_secret(
        self,
        user_id: str,
        server_id: str,
        auth_type: str,
    ) -> Optional[Dict[str, Any]]:
        rec = self.repo.get(user_id=user_id, server_id=server_id, auth_type=auth_type)
        if rec is None:
            return None
        return self.decrypt_dict(rec.encrypted_payload)

    def store_bearer_token(
        self,
        user_id: str,
        server_id: str,
        token: str,
        header_name: str = "Authorization",
    ) -> EncryptedCredentialRecord:
        cleaned = (token or "").strip()
        if not cleaned:
            raise MCPSecurityError("Bearer token cannot be empty.")
        return self.store_secret(
            user_id=user_id,
            server_id=server_id,
            auth_type="bearer_token",
            secret_payload={"token": cleaned},
            header_name=header_name or "Authorization",
        )

    def store_api_key(
        self,
        user_id: str,
        server_id: str,
        api_key: str,
        header_name: str = "X-API-Key",
    ) -> EncryptedCredentialRecord:
        cleaned = (api_key or "").strip()
        if not cleaned:
            raise MCPSecurityError("API key cannot be empty.")
        return self.store_secret(
            user_id=user_id,
            server_id=server_id,
            auth_type="api_key",
            secret_payload={"api_key": cleaned},
            header_name=header_name or "X-API-Key",
        )

    def store_env_secrets(
        self,
        user_id: str,
        server_id: str,
        secrets_map: Mapping[str, str],
    ) -> Optional[EncryptedCredentialRecord]:
        cleaned = {
            str(k).strip(): str(v).strip()
            for k, v in (secrets_map or {}).items()
            if str(k).strip() and str(v).strip()
        }
        if not cleaned:
            return None
        return self.store_secret(
            user_id=user_id,
            server_id=server_id,
            auth_type="env_secrets",
            secret_payload=cleaned,
            header_name="ENV",
        )

    def load_env_secrets(self, user_id: str, server_id: str) -> Dict[str, str]:
        data = self.load_secret(user_id=user_id, server_id=server_id, auth_type="env_secrets") or {}
        return {str(k): str(v) for k, v in data.items() if str(k).strip() and str(v)}

    def delete_credentials(
        self,
        user_id: str,
        server_id: str,
        auth_type: Optional[str] = None,
    ) -> int:
        return self.repo.delete(user_id=user_id, server_id=server_id, auth_type=auth_type)

    def get_credential_status(self, user_id: str, server_id: str) -> Dict[str, Any]:
        """Return strictly non-secret credential metadata safe for UI rendering and logs."""
        records = self.repo.list_for_server(user_id=user_id, server_id=server_id)
        if not records:
            return {
                "configured": False,
                "auth_types": [],
                "expires_at": None,
                "expired": False,
                "has_refresh_token": False,
                "scopes": "",
            }

        now = time.time()
        oauth_rec = next((r for r in records if r.auth_type == "oauth_token"), None)
        expired = bool(oauth_rec and oauth_rec.expires_at and oauth_rec.expires_at <= now)
        has_refresh = False
        if oauth_rec is not None:
            try:
                tok = self.decrypt_dict(oauth_rec.encrypted_payload)
                has_refresh = bool(tok.get("refresh_token"))
            except MCPSecurityError:
                has_refresh = False

        return {
            "configured": True,
            "auth_types": sorted({r.auth_type for r in records}),
            "expires_at": oauth_rec.expires_at if oauth_rec else None,
            "expired": expired,
            "has_refresh_token": has_refresh,
            "scopes": oauth_rec.scopes if oauth_rec else "",
            "updated_at": max(r.updated_at for r in records),
        }


class EncryptedSQLiteTokenStorage(TokenStorage):
    """MCP SDK `TokenStorage` adapter persisting OAuth tokens & client metadata via Fernet."""

    def __init__(self, vault: CredentialVault, user_id: str, server_id: str) -> None:
        self.vault = vault
        self.user_id = user_id
        self.server_id = server_id

    async def get_tokens(self) -> Optional[OAuthToken]:
        data = self.vault.load_secret(self.user_id, self.server_id, "oauth_token")
        if not data:
            return None
        try:
            return OAuthToken.model_validate(data)
        except Exception:
            return None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        expires_at = (
            time.time() + int(tokens.expires_in)
            if tokens.expires_in is not None and tokens.expires_in > 0
            else None
        )
        self.vault.store_secret(
            user_id=self.user_id,
            server_id=self.server_id,
            auth_type="oauth_token",
            secret_payload=tokens.model_dump(mode="json", exclude_none=True),
            header_name="Authorization",
            expires_at=expires_at,
            scopes=tokens.scope or "",
        )

    async def get_client_info(self) -> Optional[OAuthClientInformationFull]:
        data = self.vault.load_secret(self.user_id, self.server_id, "oauth_client")
        if not data:
            return None
        try:
            return OAuthClientInformationFull.model_validate(data)
        except Exception:
            return None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self.vault.store_secret(
            user_id=self.user_id,
            server_id=self.server_id,
            auth_type="oauth_client",
            secret_payload=client_info.model_dump(mode="json", exclude_none=True),
            header_name="Authorization",
        )
