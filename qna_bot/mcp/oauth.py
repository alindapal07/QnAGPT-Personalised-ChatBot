"""OAuth 2.1 Authorization Code + PKCE coordinator, metadata discovery, refresh & revocation."""

from dataclasses import dataclass, field
import secrets
import threading
import time
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

import httpx2
from mcp.client.auth.oauth2 import OAuthClientProvider, PKCEParameters
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken

from qna_bot.core.logging_config import get_logger
from qna_bot.mcp.credentials import CredentialVault, EncryptedSQLiteTokenStorage
from qna_bot.mcp.exceptions import MCPConnectionError, MCPSecurityError
from qna_bot.mcp.security import validate_remote_url

logger = get_logger(__name__)

_OAUTH_PENDING_TTL_SEC = 600


@dataclass
class PendingOAuthSession:
    """In-memory server-side state binding PKCE verifier & CSRF state to `(user_id, server_id)`."""

    state: str
    user_id: str
    server_id: str
    server_url: str
    authorization_endpoint: str
    token_endpoint: str
    revocation_endpoint: str = ""
    registration_endpoint: str = ""
    client_id: str = ""
    client_secret: str = ""
    redirect_uri: str = "http://localhost:8501/oauth/callback"
    scopes: str = ""
    code_verifier: str = ""
    authorization_url: str = ""
    allow_insecure_http: bool = False
    allow_private_network: bool = False
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + _OAUTH_PENDING_TTL_SEC)

    def safe_public_view(self) -> Dict[str, Any]:
        """Return UI-safe OAuth state summary omitting PKCE verifier & client_secret."""
        return {
            "state": self.state,
            "server_id": self.server_id,
            "authorization_endpoint": self.authorization_endpoint,
            "token_endpoint": self.token_endpoint,
            "authorization_url": self.authorization_url,
            "redirect_uri": self.redirect_uri,
            "scopes": self.scopes,
            "expires_at": self.expires_at,
        }


class OAuthFlowManager:
    """Coordinates PKCE authorization, token exchange, refresh, revocation, and SSRF validation."""

    def __init__(
        self,
        vault: CredentialVault,
        default_redirect_uri: str = "http://localhost:8501/oauth/callback",
    ) -> None:
        self.vault = vault
        self.default_redirect_uri = default_redirect_uri
        self._lock = threading.RLock()
        self._pending_by_state: Dict[str, PendingOAuthSession] = {}

    def _cleanup_expired(self) -> None:
        now = time.time()
        expired = [st for st, sess in self._pending_by_state.items() if sess.expires_at <= now]
        for st in expired:
            self._pending_by_state.pop(st, None)

    def get_pending_for_server(self, user_id: str, server_id: str) -> Optional[PendingOAuthSession]:
        with self._lock:
            self._cleanup_expired()
            for sess in self._pending_by_state.values():
                if sess.user_id == user_id and sess.server_id == server_id:
                    return sess
            return None

    def discover_metadata(
        self,
        server_url: str,
        allow_insecure_http: bool = False,
        allow_private_network: bool = False,
        http_get: Optional[Callable[[str], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Discover Protected Resource & OAuth Authorization Server metadata with SSRF validation."""
        validated_server = validate_remote_url(
            server_url,
            allow_insecure_http=allow_insecure_http,
            allow_private_network=allow_private_network,
        )
        parsed = urlparse(validated_server)
        origin = f"{parsed.scheme}://{parsed.netloc}"

        def _fetch_json(target_url: str) -> Dict[str, Any]:
            validate_remote_url(
                target_url,
                allow_insecure_http=allow_insecure_http,
                allow_private_network=allow_private_network,
            )
            if http_get is not None:
                return http_get(target_url)
            with httpx2.Client(timeout=8.0, follow_redirects=False) as client:
                resp = client.get(target_url, headers={"Accept": "application/json"})
                if resp.status_code == 200:
                    data = resp.json()
                    return data if isinstance(data, dict) else {}
            return {}

        auth_server_origin = origin
        prm_candidates = [
            urljoin(origin, f"/.well-known/oauth-protected-resource{parsed.path.rstrip('/')}"),
            urljoin(origin, "/.well-known/oauth-protected-resource"),
        ]
        prm: Dict[str, Any] = {}
        for url in prm_candidates:
            try:
                prm = _fetch_json(url)
                if prm:
                    break
            except Exception:
                continue

        auth_servers = prm.get("authorization_servers") or []
        if isinstance(auth_servers, list) and auth_servers:
            candidate_as = str(auth_servers[0]).strip()
            validate_remote_url(
                candidate_as,
                allow_insecure_http=allow_insecure_http,
                allow_private_network=allow_private_network,
            )
            auth_server_origin = candidate_as.rstrip("/")

        as_meta: Dict[str, Any] = {}
        for well_known in (
            f"{auth_server_origin}/.well-known/oauth-authorization-server",
            f"{auth_server_origin}/.well-known/openid-configuration",
        ):
            try:
                as_meta = _fetch_json(well_known)
                if as_meta:
                    break
            except Exception:
                continue

        auth_endpoint = str(as_meta.get("authorization_endpoint") or urljoin(origin, "/authorize"))
        token_endpoint = str(as_meta.get("token_endpoint") or urljoin(origin, "/token"))
        registration_endpoint = str(as_meta.get("registration_endpoint") or "")
        revocation_endpoint = str(as_meta.get("revocation_endpoint") or "")

        # Validate every discovered endpoint against SSRF protections
        validate_remote_url(
            auth_endpoint,
            allow_insecure_http=allow_insecure_http,
            allow_private_network=allow_private_network,
        )
        validate_remote_url(
            token_endpoint,
            allow_insecure_http=allow_insecure_http,
            allow_private_network=allow_private_network,
        )
        if registration_endpoint:
            validate_remote_url(
                registration_endpoint,
                allow_insecure_http=allow_insecure_http,
                allow_private_network=allow_private_network,
            )
        if revocation_endpoint:
            validate_remote_url(
                revocation_endpoint,
                allow_insecure_http=allow_insecure_http,
                allow_private_network=allow_private_network,
            )

        scopes_supported = as_meta.get("scopes_supported") or prm.get("scopes_supported") or []
        return {
            "authorization_endpoint": auth_endpoint,
            "token_endpoint": token_endpoint,
            "registration_endpoint": registration_endpoint,
            "revocation_endpoint": revocation_endpoint,
            "scopes_supported": list(scopes_supported) if isinstance(scopes_supported, list) else [],
            "discovered": bool(as_meta or prm),
        }

    def start_authorization_flow(
        self,
        user_id: str,
        server_id: str,
        server_url: str,
        client_id: str = "",
        client_secret: str = "",
        authorization_endpoint: str = "",
        token_endpoint: str = "",
        revocation_endpoint: str = "",
        redirect_uri: str = "",
        scopes: str = "",
        allow_insecure_http: bool = False,
        allow_private_network: bool = False,
    ) -> PendingOAuthSession:
        """Initiate PKCE + CSRF state OAuth flow bound strictly to `(user_id, server_id)`."""
        validate_remote_url(
            server_url,
            allow_insecure_http=allow_insecure_http,
            allow_private_network=allow_private_network,
        )
        auth_ep = (authorization_endpoint or "").strip()
        tok_ep = (token_endpoint or "").strip()
        rev_ep = (revocation_endpoint or "").strip()
        reg_ep = ""

        if not auth_ep or not tok_ep:
            discovered = self.discover_metadata(
                server_url,
                allow_insecure_http=allow_insecure_http,
                allow_private_network=allow_private_network,
            )
            auth_ep = auth_ep or discovered["authorization_endpoint"]
            tok_ep = tok_ep or discovered["token_endpoint"]
            rev_ep = rev_ep or discovered["revocation_endpoint"]
            reg_ep = discovered.get("registration_endpoint", "")
            if not scopes and discovered["scopes_supported"]:
                scopes = " ".join(str(s) for s in discovered["scopes_supported"])

        validate_remote_url(
            auth_ep,
            allow_insecure_http=allow_insecure_http,
            allow_private_network=allow_private_network,
        )
        validate_remote_url(
            tok_ep,
            allow_insecure_http=allow_insecure_http,
            allow_private_network=allow_private_network,
        )
        if rev_ep:
            validate_remote_url(
                rev_ep,
                allow_insecure_http=allow_insecure_http,
                allow_private_network=allow_private_network,
            )

        resolved_redirect = (redirect_uri or self.default_redirect_uri).strip()
        resolved_client_id = client_id.strip() or "qnagpt-mcp-client"
        resolved_secret = client_secret.strip()

        pkce = PKCEParameters.generate()
        state = secrets.token_urlsafe(32)
        params: Dict[str, str] = {
            "response_type": "code",
            "client_id": resolved_client_id,
            "redirect_uri": resolved_redirect,
            "state": state,
            "code_challenge": pkce.code_challenge,
            "code_challenge_method": "S256",
            "resource": server_url.rstrip("/"),
        }
        if scopes.strip():
            params["scope"] = scopes.strip()

        sep = "&" if "?" in auth_ep else "?"
        auth_url = f"{auth_ep}{sep}{urlencode(params)}"

        session = PendingOAuthSession(
            state=state,
            user_id=user_id,
            server_id=server_id,
            server_url=server_url,
            authorization_endpoint=auth_ep,
            token_endpoint=tok_ep,
            revocation_endpoint=rev_ep,
            registration_endpoint=reg_ep,
            client_id=resolved_client_id,
            client_secret=resolved_secret,
            redirect_uri=resolved_redirect,
            scopes=scopes.strip(),
            code_verifier=pkce.code_verifier,
            authorization_url=auth_url,
            allow_insecure_http=allow_insecure_http,
            allow_private_network=allow_private_network,
        )

        # Persist client configuration encrypted alongside server credentials
        self.vault.store_secret(
            user_id=user_id,
            server_id=server_id,
            auth_type="oauth_client",
            secret_payload={
                "client_id": resolved_client_id,
                "client_secret": resolved_secret or None,
                "redirect_uris": [resolved_redirect],
                "token_endpoint_auth_method": "client_secret_post" if resolved_secret else "none",
                "token_endpoint": tok_ep,
                "revocation_endpoint": rev_ep,
                "authorization_endpoint": auth_ep,
            },
            header_name="Authorization",
        )

        with self._lock:
            self._cleanup_expired()
            # Replace any stale pending session for same (user_id, server_id)
            for k in [
                k for k, v in self._pending_by_state.items()
                if v.user_id == user_id and v.server_id == server_id
            ]:
                self._pending_by_state.pop(k, None)
            self._pending_by_state[state] = session

        return session

    def complete_authorization_callback(
        self,
        user_id: str,
        server_id: str,
        callback_url_or_code: str,
        state: Optional[str] = None,
        http_post: Optional[Callable[[str, Dict[str, str]], Dict[str, Any]]] = None,
    ) -> OAuthToken:
        """Validate OAuth state & exchange authorization code + PKCE verifier for encrypted tokens."""
        raw = (callback_url_or_code or "").strip()
        if not raw:
            raise MCPSecurityError("OAuth callback URL or authorization code is required.")

        parsed_code = raw
        parsed_state = (state or "").strip()
        if raw.startswith(("http://", "https://")) or ("code=" in raw and "state=" in raw):
            parsed = urlparse(raw if "://" in raw else f"https://localhost/cb?{raw.lstrip('?')}")
            qs = parse_qs(parsed.query)
            if qs.get("error"):
                err_desc = qs.get("error_description", [qs["error"][0]])[0]
                raise MCPSecurityError(f"OAuth authorization rejected: {err_desc}")
            parsed_code = (qs.get("code") or [""])[0].strip()
            if not parsed_state:
                parsed_state = (qs.get("state") or [""])[0].strip()

        if not parsed_code:
            raise MCPSecurityError("Authorization code missing from OAuth callback.")

        with self._lock:
            self._cleanup_expired()
            pending = self.get_pending_for_server(user_id=user_id, server_id=server_id)
            if pending is None:
                raise MCPSecurityError("No active pending OAuth authorization session found or session expired.")
            if not parsed_state or not secrets.compare_digest(parsed_state, pending.state):
                raise MCPSecurityError("OAuth state validation failed (CSRF state mismatch).")
            self._pending_by_state.pop(pending.state, None)

        validate_remote_url(
            pending.token_endpoint,
            allow_insecure_http=pending.allow_insecure_http,
            allow_private_network=pending.allow_private_network,
        )
        form = {
            "grant_type": "authorization_code",
            "code": parsed_code,
            "redirect_uri": pending.redirect_uri,
            "client_id": pending.client_id,
            "code_verifier": pending.code_verifier,
        }
        if pending.client_secret:
            form["client_secret"] = pending.client_secret

        token_json = self._post_token_form(pending.token_endpoint, form, http_post=http_post)
        return self._persist_oauth_token_payload(
            user_id=user_id,
            server_id=server_id,
            token_json=token_json,
            fallback_scope=pending.scopes,
        )

    def refresh_access_token(
        self,
        user_id: str,
        server_id: str,
        allow_insecure_http: bool = False,
        allow_private_network: bool = False,
        http_post: Optional[Callable[[str, Dict[str, str]], Dict[str, Any]]] = None,
    ) -> OAuthToken:
        """Refresh a stored OAuth token cleanly and re-encrypt in the CredentialVault."""
        token_data = self.vault.load_secret(user_id, server_id, "oauth_token") or {}
        client_data = self.vault.load_secret(user_id, server_id, "oauth_client") or {}
        refresh_tok = str(token_data.get("refresh_token") or "").strip()
        if not refresh_tok:
            raise MCPConnectionError("No OAuth refresh_token available; please re-authorize.")

        token_endpoint = str(client_data.get("token_endpoint") or "").strip()
        if not token_endpoint:
            raise MCPConnectionError("OAuth token_endpoint is not recorded for this server.")

        validate_remote_url(
            token_endpoint,
            allow_insecure_http=allow_insecure_http,
            allow_private_network=allow_private_network,
        )
        form: Dict[str, str] = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_tok,
            "client_id": str(client_data.get("client_id") or "qnagpt-mcp-client"),
        }
        if client_data.get("client_secret"):
            form["client_secret"] = str(client_data["client_secret"])

        refreshed = self._post_token_form(token_endpoint, form, http_post=http_post)
        if not refreshed.get("refresh_token"):
            refreshed["refresh_token"] = refresh_tok
        return self._persist_oauth_token_payload(
            user_id=user_id,
            server_id=server_id,
            token_json=refreshed,
            fallback_scope=str(token_data.get("scope") or ""),
        )

    def revoke_access(
        self,
        user_id: str,
        server_id: str,
        allow_insecure_http: bool = False,
        allow_private_network: bool = False,
        http_post: Optional[Callable[[str, Dict[str, str]], Dict[str, Any]]] = None,
    ) -> bool:
        """Attempt remote RFC 7009 revocation if supported and delete encrypted local credentials."""
        revoked_remotely = False
        token_data = self.vault.load_secret(user_id, server_id, "oauth_token") or {}
        client_data = self.vault.load_secret(user_id, server_id, "oauth_client") or {}
        revocation_ep = str(client_data.get("revocation_endpoint") or "").strip()
        access_tok = str(token_data.get("access_token") or "").strip()

        if revocation_ep and access_tok:
            try:
                validate_remote_url(
                    revocation_ep,
                    allow_insecure_http=allow_insecure_http,
                    allow_private_network=allow_private_network,
                )
                form: Dict[str, str] = {"token": access_tok}
                if client_data.get("client_id"):
                    form["client_id"] = str(client_data["client_id"])
                if http_post is not None:
                    http_post(revocation_ep, form)
                    revoked_remotely = True
                else:
                    with httpx2.Client(timeout=6.0, follow_redirects=False) as client:
                        resp = client.post(revocation_ep, data=form)
                        revoked_remotely = resp.status_code in (200, 204)
            except Exception as exc:
                logger.warning("Remote OAuth revocation failed cleanly for %s: %s", server_id, exc)

        deleted = self.vault.delete_credentials(user_id=user_id, server_id=server_id)
        with self._lock:
            for k in [
                k for k, v in self._pending_by_state.items()
                if v.user_id == user_id and v.server_id == server_id
            ]:
                self._pending_by_state.pop(k, None)
        return revoked_remotely or deleted > 0

    def create_sdk_auth_provider(
        self,
        user_id: str,
        server_id: str,
        server_url: str,
        allow_insecure_http: bool = False,
        allow_private_network: bool = False,
        redirect_handler: Optional[Callable[[str], Awaitable[None]]] = None,
        callback_handler: Optional[Callable[[], Awaitable[Any]]] = None,
    ) -> OAuthClientProvider:
        """Create official MCP SDK `OAuthClientProvider` using encrypted SQLite token storage."""
        storage = EncryptedSQLiteTokenStorage(self.vault, user_id=user_id, server_id=server_id)
        client_meta = OAuthClientMetadata(
            redirect_uris=[self.default_redirect_uri],
            client_name="QnAGPT Agentic Workspace",
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
        )

        async def _validate_resource(srv: str, prm_res: Optional[str]) -> None:
            validate_remote_url(
                srv,
                allow_insecure_http=allow_insecure_http,
                allow_private_network=allow_private_network,
            )
            if prm_res:
                validate_remote_url(
                    prm_res,
                    allow_insecure_http=allow_insecure_http,
                    allow_private_network=allow_private_network,
                )

        async def _default_redirect(url: str) -> None:
            raise MCPConnectionError(
                "OAuth authorization is required. Complete sign-in in the Connect panel first."
            )

        async def _default_callback() -> Any:
            raise MCPConnectionError("OAuth callback handler requires interactive browser approval.")

        return OAuthClientProvider(
            server_url=server_url,
            client_metadata=client_meta,
            storage=storage,
            redirect_handler=redirect_handler or _default_redirect,
            callback_handler=callback_handler or _default_callback,
            validate_resource_url=_validate_resource,
        )

    def _post_token_form(
        self,
        token_endpoint: str,
        form: Dict[str, str],
        http_post: Optional[Callable[[str, Dict[str, str]], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if http_post is not None:
            data = http_post(token_endpoint, form)
        else:
            with httpx2.Client(timeout=10.0, follow_redirects=False) as client:
                resp = client.post(
                    token_endpoint,
                    data=form,
                    headers={"Accept": "application/json"},
                )
                if resp.status_code >= 400:
                    raise MCPConnectionError(
                        f"OAuth token request failed ({resp.status_code})."
                    )
                data = resp.json()

        if not isinstance(data, dict) or not data.get("access_token"):
            err = data.get("error_description") or data.get("error") if isinstance(data, dict) else "invalid payload"
            raise MCPConnectionError(f"OAuth token exchange did not return access_token: {err}")
        return data

    def _persist_oauth_token_payload(
        self,
        user_id: str,
        server_id: str,
        token_json: Mapping[str, Any],
        fallback_scope: str = "",
    ) -> OAuthToken:
        expires_in_raw = token_json.get("expires_in")
        expires_in: Optional[int] = None
        if expires_in_raw is not None:
            try:
                expires_in = int(expires_in_raw)
            except (TypeError, ValueError):
                expires_in = None

        token = OAuthToken(
            access_token=str(token_json["access_token"]),
            token_type=str(token_json.get("token_type") or "Bearer"),
            expires_in=expires_in,
            scope=str(token_json.get("scope") or fallback_scope or ""),
            refresh_token=str(token_json["refresh_token"]) if token_json.get("refresh_token") else None,
        )
        expires_at = time.time() + expires_in if expires_in and expires_in > 0 else None
        self.vault.store_secret(
            user_id=user_id,
            server_id=server_id,
            auth_type="oauth_token",
            secret_payload=token.model_dump(mode="json", exclude_none=True),
            header_name="Authorization",
            expires_at=expires_at,
            scopes=token.scope or "",
        )
        return token
