"""Shared HTTP client, TTL caching, and rate-limit handling for MCP Registry providers."""

import json
import time
from typing import Any, Callable, Dict, Optional, Protocol
from urllib.parse import urlencode

import httpx2

from qna_bot.core.logging_config import get_logger
from qna_bot.database.repositories import RegistryCacheRepository
from qna_bot.mcp.exceptions import MCPError
from qna_bot.mcp.registry.models import RegistryListing, RegistryPage
from qna_bot.mcp.security import validate_remote_url

logger = get_logger(__name__)


class RegistryError(MCPError):
    """Raised when an MCP Registry API call fails."""


class RegistryRateLimitError(RegistryError):
    """Raised when an MCP Registry API returns HTTP 429 rate limiting."""


class MCPRegistryProvider(Protocol):
    """Protocol implemented by independent MCP registry directory providers."""

    source_id: str
    display_name: str
    attribution_url: str

    def list_servers(
        self,
        query: str = "",
        limit: int = 24,
        cursor: Optional[str] = None,
        sort: str = "all",
    ) -> RegistryPage: ...

    def get_server_details(self, registry_id: str) -> Optional[RegistryListing]: ...


class CachedRegistryHTTPClient:
    """Rate-limit-aware HTTP GET helper backed by SQLite TTL caching and SSRF URL checks."""

    def __init__(
        self,
        cache_repo: Optional[RegistryCacheRepository] = None,
        cache_ttl_sec: int = 300,
        timeout_sec: float = 12.0,
        transport: Optional[Callable[[str, Dict[str, str]], tuple[int, Dict[str, str], str]]] = None,
    ) -> None:
        self.cache_repo = cache_repo
        self.cache_ttl_sec = max(10, int(cache_ttl_sec))
        self.timeout_sec = timeout_sec
        self._transport = transport

    def get_json(
        self,
        source: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        use_cache: bool = True,
        allow_insecure_http: bool = False,
        allow_private_network: bool = False,
    ) -> tuple[Dict[str, Any], bool]:
        """Fetch JSON payload from a validated URL with SQLite caching and 429 Retry-After support."""
        clean_params = {k: v for k, v in (params or {}).items() if v is not None and v != ""}
        full_url = f"{url}?{urlencode(clean_params)}" if clean_params else url
        validate_remote_url(
            full_url,
            allow_insecure_http=allow_insecure_http,
            allow_private_network=allow_private_network,
        )

        cache_key = f"{source}:{full_url}"
        if use_cache and self.cache_repo is not None:
            cached = self.cache_repo.get(cache_key)
            if cached is not None:
                payload_json, _ = cached
                try:
                    parsed = json.loads(payload_json)
                    if isinstance(parsed, dict):
                        return parsed, True
                except ValueError:
                    pass

        req_headers = {"Accept": "application/json", **(headers or {})}

        for attempt in range(2):
            status_code, resp_headers, body_text = self._perform_get(full_url, req_headers)
            if status_code == 429:
                retry_raw = resp_headers.get("retry-after") or resp_headers.get("Retry-After") or "1"
                try:
                    wait_sec = min(2.0, max(0.2, float(retry_raw)))
                except ValueError:
                    wait_sec = 1.0
                if attempt == 0:
                    time.sleep(wait_sec)
                    continue
                raise RegistryRateLimitError(
                    f"{source} registry rate limit exceeded (HTTP 429). Please try again shortly."
                )

            if status_code in (401, 403):
                raise RegistryError(f"{source} registry authentication failed (HTTP {status_code}).")

            if status_code == 404:
                return {}, False

            if status_code >= 400:
                raise RegistryError(
                    f"{source} registry returned HTTP {status_code}."
                )

            try:
                data = json.loads(body_text) if body_text else {}
            except ValueError as exc:
                raise RegistryError(f"{source} registry returned invalid JSON: {exc}") from exc

            if not isinstance(data, dict):
                data = {"items": data} if isinstance(data, list) else {}

            if use_cache and self.cache_repo is not None:
                self.cache_repo.put(
                    cache_key=cache_key,
                    source=source,
                    payload_json=json.dumps(data),
                    ttl_sec=self.cache_ttl_sec,
                )
            return data, False

        raise RegistryError(f"{source} registry request failed after retries.")

    def _perform_get(self, url: str, headers: Dict[str, str]) -> tuple[int, Dict[str, str], str]:
        if self._transport is not None:
            return self._transport(url, headers)
        try:
            with httpx2.Client(timeout=self.timeout_sec, follow_redirects=False) as client:
                resp = client.get(url, headers=headers)
                return resp.status_code, dict(resp.headers), resp.text
        except httpx2.HTTPError as exc:
            raise RegistryError(f"Network error contacting registry: {exc}") from exc
