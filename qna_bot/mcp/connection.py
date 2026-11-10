"""MCP transport lifecycle, background event loop runner, and auth header resolution."""

import asyncio
from contextlib import AsyncExitStack
import inspect
import os
import threading
from typing import Any, Awaitable, Callable, Dict, Optional, Protocol, TypeVar

import httpx2
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

from qna_bot.core.logging_config import get_logger
from qna_bot.database.models import MCPServerRecord
from qna_bot.mcp.config import DEFAULT_CONNECT_TIMEOUT_SEC, DEFAULT_TOOL_TIMEOUT_SEC
from qna_bot.mcp.credentials import CredentialVault
from qna_bot.mcp.exceptions import MCPConnectionError, MCPSecurityError
from qna_bot.mcp.oauth import OAuthFlowManager
from qna_bot.mcp.security import validate_remote_url, validate_stdio_config

logger = get_logger(__name__)
_T = TypeVar("_T")


def _format_exception_detail(exc: BaseException) -> str:
    """Extract leaf error messages cleanly out of nested AnyIO/Python BaseExceptionGroups."""
    if hasattr(exc, "exceptions") and getattr(exc, "exceptions", None):
        leaves: list[str] = []
        for sub in getattr(exc, "exceptions"):
            msg = _format_exception_detail(sub)
            if msg and msg not in leaves:
                leaves.append(msg)
        if leaves:
            return "; ".join(leaves)
    return str(exc) or type(exc).__name__


class MCPSessionLike(Protocol):
    """Minimal session protocol implemented by mcp.client.session.ClientSession and test doubles."""

    async def list_tools(self) -> Any: ...
    async def list_resources(self) -> Any: ...
    async def list_prompts(self) -> Any: ...
    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any: ...
    async def read_resource(self, uri: str) -> Any: ...
    async def get_prompt(self, name: str, arguments: Optional[Dict[str, str]] = None) -> Any: ...
    async def send_ping(self) -> Any: ...


class AsyncLoopRunner:
    """Dedicated single background event loop thread for async MCP client sessions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()

    def _ensure_started(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is not None and not self._loop.is_closed():
                return self._loop

            self._ready.clear()

            def _worker() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._loop = loop
                self._ready.set()
                try:
                    loop.run_forever()
                finally:
                    pending = asyncio.all_tasks(loop)
                    for task in pending:
                        task.cancel()
                    if pending:
                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    loop.close()

            self._thread = threading.Thread(target=_worker, name="qnabot-mcp-loop", daemon=True)
            self._thread.start()
            self._ready.wait(timeout=5.0)
            if self._loop is None:
                raise MCPConnectionError("Failed to start background MCP async loop.")
            return self._loop

    def run(self, coro: Awaitable[_T], timeout: float = 30.0) -> _T:
        try:
            loop = self._ensure_started()
        except BaseException:
            if inspect.iscoroutine(coro):
                coro.close()
            raise
        future = asyncio.run_coroutine_threadsafe(coro, loop)  # type: ignore[arg-type]
        return future.result(timeout=timeout)

    def close(self) -> None:
        with self._lock:
            if self._loop is not None and not self._loop.is_closed():
                self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=3.0)
            self._loop = None
            self._thread = None


def resolve_auth_headers(
    record: MCPServerRecord,
    vault: Optional[CredentialVault] = None,
) -> Dict[str, str]:
    """Resolve remote authentication headers from encrypted vault or environment variables."""
    auth_type = (record.auth_type or "none").lower()
    if auth_type == "none":
        return {}

    header_name = (record.auth_header_name or "Authorization").strip()

    if auth_type == "bearer_token":
        if vault is None:
            raise MCPConnectionError("Encrypted credential vault is required for bearer_token auth.")
        secret = vault.load_secret(record.user_id, record.server_id, "bearer_token") or {}
        tok = str(secret.get("token") or "").strip()
        if not tok:
            raise MCPConnectionError(
                f"Bearer token credential for '{record.name}' is missing. Please provide your token to connect."
            )
        value = tok if tok.lower().startswith("bearer ") else f"Bearer {tok}"
        return {header_name: value}

    if auth_type == "api_key":
        if vault is None:
            raise MCPConnectionError("Encrypted credential vault is required for api_key auth.")
        secret = vault.load_secret(record.user_id, record.server_id, "api_key") or {}
        key_val = str(secret.get("api_key") or "").strip()
        if not key_val:
            raise MCPConnectionError(
                f"API key credential for '{record.name}' is missing. Please provide your API key to connect."
            )
        hdr = (record.auth_header_name or "X-API-Key").strip() or "X-API-Key"
        return {hdr: key_val}

    if auth_type == "oauth":
        if vault is not None:
            oauth_data = vault.load_secret(record.user_id, record.server_id, "oauth_token") or {}
            tok = str(oauth_data.get("access_token") or "").strip()
            if tok:
                tok_type = str(oauth_data.get("token_type") or "Bearer").strip() or "Bearer"
                return {header_name: f"{tok_type} {tok}"}
        env_var = (record.auth_env_var or "").strip()
        if env_var and os.getenv(env_var, "").strip():
            tok = os.getenv(env_var, "").strip()
            value = tok if tok.lower().startswith("bearer ") else f"Bearer {tok}"
            return {header_name: value}
        raise MCPConnectionError(
            f"OAuth authorization is required for '{record.name}'. Complete OAuth sign-in before connecting."
        )

    env_var = (record.auth_env_var or "").strip()
    if not env_var:
        raise MCPSecurityError(
            f"Server '{record.name}' requires auth_env_var for auth_type='{auth_type}'."
        )
    secret_val = os.getenv(env_var, "").strip()
    if not secret_val:
        raise MCPConnectionError(
            f"Environment variable '{env_var}' required by MCP server '{record.name}' is not set."
        )

    if auth_type == "bearer_env":
        value = secret_val if secret_val.lower().startswith("bearer ") else f"Bearer {secret_val}"
        return {header_name: value}
    if auth_type == "api_key_env":
        return {header_name: secret_val}

    raise MCPSecurityError(f"Unsupported auth_type '{auth_type}'.")


class MCPConnectionHandle:
    """Manages a single live MCP ClientSession in a dedicated task to preserve AnyIO cancel scopes."""

    def __init__(
        self,
        record: MCPServerRecord,
        loop_runner: AsyncLoopRunner,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_SEC,
        tool_timeout: float = DEFAULT_TOOL_TIMEOUT_SEC,
        session_factory: Optional[Callable[[MCPServerRecord, AsyncExitStack], Awaitable[Any]]] = None,
        vault: Optional[CredentialVault] = None,
        oauth_manager: Optional[OAuthFlowManager] = None,
        max_retries: int = 1,
    ) -> None:
        self.record = record
        self._runner = loop_runner
        self._connect_timeout = connect_timeout
        self._tool_timeout = float(tool_timeout)
        self._session_factory = session_factory
        self._vault = vault
        self._oauth_manager = oauth_manager
        self._max_retries = max(0, int(max_retries))
        self._session: Optional[Any] = None
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_event: Optional[asyncio.Event] = None

    def configure_auth(
        self,
        vault: Optional[CredentialVault],
        oauth_manager: Optional[OAuthFlowManager],
    ) -> None:
        self._vault = vault
        self._oauth_manager = oauth_manager

    def configure_timeouts(self, tool_timeout: float) -> None:
        self._tool_timeout = float(tool_timeout)

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    @property
    def session(self) -> Any:
        if self._session is None:
            raise MCPConnectionError(f"MCP server '{self.record.name}' is not connected.")
        return self._session

    def _resolve_remote_auth(
        self,
        url: str,
    ) -> tuple[Dict[str, str], Optional[httpx2.Auth]]:
        auth_type = (self.record.auth_type or "none").lower()
        if (
            auth_type == "oauth"
            and self._oauth_manager is not None
            and self._vault is not None
            and self._vault.load_secret(self.record.user_id, self.record.server_id, "oauth_client")
        ):
            # Also verify tokens exist or refreshable client info exists
            tok = self._vault.load_secret(self.record.user_id, self.record.server_id, "oauth_token")
            if not tok:
                raise MCPConnectionError(
                    f"OAuth authorization is required for '{self.record.name}'. Complete OAuth sign-in first."
                )
            provider = self._oauth_manager.create_sdk_auth_provider(
                user_id=self.record.user_id,
                server_id=self.record.server_id,
                server_url=url,
                allow_insecure_http=self.record.allow_insecure_http,
                allow_private_network=self.record.allow_private_network,
            )
            return {}, provider
        return resolve_auth_headers(self.record, vault=self._vault), None

    async def _open_transport_session(self, stack: AsyncExitStack) -> Any:
        if self._session_factory is not None:
            return await self._session_factory(self.record, stack)

        transport = (self.record.transport or "").lower()
        if transport == "stdio":
            cmd, args, safe_env = validate_stdio_config(
                command=self.record.command,
                args=self.record.args,
                env_refs=self.record.env_refs,
                allowed_roots=self.record.allowed_roots,
            )
            if self._vault is not None:
                vault_env = self._vault.load_env_secrets(self.record.user_id, self.record.server_id)
                for k, v in vault_env.items():
                    if k.strip() and v.strip():
                        safe_env[k.strip()] = v.strip()

            params = StdioServerParameters(command=cmd, args=args, env=safe_env)
            streams = await stack.enter_async_context(stdio_client(params))
        elif transport == "streamable_http":
            url = validate_remote_url(
                self.record.url,
                allow_insecure_http=self.record.allow_insecure_http,
                allow_private_network=self.record.allow_private_network,
            )
            headers, sdk_auth = self._resolve_remote_auth(url)
            http_client = await stack.enter_async_context(
                create_mcp_http_client(
                    headers=headers or None,
                    timeout=httpx2.Timeout(self._connect_timeout, read=max(90.0, self._tool_timeout + 15.0)),
                    auth=sdk_auth,
                )
            )
            streams = await stack.enter_async_context(
                streamable_http_client(url, http_client=http_client)
            )
        elif transport == "sse":
            url = validate_remote_url(
                self.record.url,
                allow_insecure_http=self.record.allow_insecure_http,
                allow_private_network=self.record.allow_private_network,
            )
            headers, sdk_auth = self._resolve_remote_auth(url)
            streams = await stack.enter_async_context(
                sse_client(
                    url,
                    headers=headers or None,
                    timeout=self._connect_timeout,
                    auth=sdk_auth,
                )
            )
        else:
            raise MCPConnectionError(f"Unsupported MCP transport '{self.record.transport}'.")

        read_stream, write_stream = streams[0], streams[1]
        session = await stack.enter_async_context(
            ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=max(float(self._connect_timeout), self._tool_timeout),
            )
        )
        await session.initialize()
        return session

    async def _connect_async(self) -> None:
        ready_event = asyncio.Event()
        stop_event = asyncio.Event()
        connect_error: list[BaseException] = []

        async def _session_lifecycle() -> None:
            try:
                async with AsyncExitStack() as stack:
                    session = await self._open_transport_session(stack)
                    self._session = session
                    ready_event.set()
                    await stop_event.wait()
            except BaseException as exc:
                if not ready_event.is_set():
                    connect_error.append(exc)
                    ready_event.set()
                else:
                    logger.warning("MCP session lifecycle error for '%s': %s", self.record.name, exc)
            finally:
                self._session = None

        self._stop_event = stop_event
        self._task = asyncio.create_task(_session_lifecycle())
        await ready_event.wait()
        if connect_error:
            await self._disconnect_async()
            raise connect_error[0]

    async def _disconnect_async(self) -> None:
        stop_event = self._stop_event
        task = self._task
        self._session = None
        self._stop_event = None
        self._task = None
        if stop_event is not None:
            stop_event.set()
        if task is not None:
            try:
                await asyncio.wait_for(task, timeout=4.0)
            except Exception:
                task.cancel()

    def connect(self) -> None:
        attempts = self._max_retries + 1
        last_exc: Optional[BaseException] = None
        for attempt in range(attempts):
            try:
                self._runner.run(self._connect_async(), timeout=self._connect_timeout + 2.0)
                return
            except MCPSecurityError:
                raise
            except Exception as exc:
                last_exc = exc
                self.disconnect()
                # Do not retry deterministic missing credential / unsupported transport errors
                if "is missing" in str(exc) or "OAuth authorization is required" in str(exc) or "Unsupported" in str(exc):
                    break
                if attempt < attempts - 1:
                    logger.info(
                        "Retrying connection to MCP server '%s' (attempt %d/%d)...",
                        self.record.name,
                        attempt + 2,
                        attempts,
                    )
                    continue
        detail = _format_exception_detail(last_exc) if last_exc is not None else "Unknown connection error"
        if isinstance(last_exc, MCPConnectionError):
            raise MCPConnectionError(detail) from last_exc
        logger.warning("Failed connecting MCP server '%s': %s", self.record.name, detail)
        raise MCPConnectionError(
            f"Failed to connect to '{self.record.name}': {detail}"
        ) from last_exc

    def disconnect(self) -> None:
        if self._task is None and self._session is None:
            return
        try:
            self._runner.run(self._disconnect_async(), timeout=5.0)
        except Exception as exc:
            logger.warning("Error closing MCP server '%s': %s", self.record.name, exc)
            self._session = None
            self._task = None
            self._stop_event = None

    def run_call(self, coro_fn: Callable[[Any], Awaitable[_T]], timeout: Optional[float] = None) -> _T:
        return self._runner.run(coro_fn(self.session), timeout=self._tool_timeout if timeout is None else timeout)
