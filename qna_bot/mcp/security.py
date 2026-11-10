"""MCP security validators for STDIO allowlisting, remote SSRF defense, and secret masking."""

import ipaddress
import os
from pathlib import Path
import re
import socket
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence
from urllib.parse import urlparse

from qna_bot.mcp.config import (
    ALLOWED_STDIO_EXECUTABLES,
    BLOCKED_PARENT_ENV_KEYS,
    MAX_TOOL_OUTPUT_CHARS,
    SENSITIVE_ARG_KEYS,
)
from qna_bot.mcp.exceptions import MCPSecurityError

_SHELL_META_RE = re.compile(r"[;&|`$><\r\n]")
_SECRET_VALUE_RE = re.compile(
    r"(Bearer\s+[A-Za-z0-9._\-]+|gsk_[A-Za-z0-9]{12,}|sk-[A-Za-z0-9\-_]{12,}|ghp_[A-Za-z0-9]{12,})",
    re.IGNORECASE,
)


def mask_secrets(value: Any) -> Any:
    """Recursively mask sensitive dictionary keys and token-shaped strings."""
    if isinstance(value, Mapping):
        masked: Dict[str, Any] = {}
        for k, v in value.items():
            key_lower = str(k).lower()
            if any(token in key_lower for token in SENSITIVE_ARG_KEYS):
                masked[str(k)] = "***REDACTED***"
            else:
                masked[str(k)] = mask_secrets(v)
        return masked
    if isinstance(value, list):
        return [mask_secrets(item) for item in value]
    if isinstance(value, str):
        return _SECRET_VALUE_RE.sub("***REDACTED***", value)
    return value


def format_masked_arguments(arguments: Mapping[str, Any]) -> str:
    """Format tool arguments safely for audit logs and approval previews."""
    import json

    return json.dumps(mask_secrets(dict(arguments or {})), sort_keys=True)


def sanitize_tool_output(text: str, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """Strip control characters, mask accidental secrets, and cap tool output size."""
    cleaned = (text or "").replace("\x00", "")
    cleaned = _SECRET_VALUE_RE.sub("***REDACTED***", cleaned)
    if len(cleaned) > max_chars:
        return cleaned[:max_chars] + f"\n... [truncated {len(cleaned) - max_chars} chars]"
    return cleaned


def validate_stdio_config(
    command: str,
    args: Sequence[str],
    env_refs: Optional[Mapping[str, str]] = None,
    allowed_roots: Optional[Sequence[str]] = None,
    extra_allowlist: Optional[Iterable[str]] = None,
) -> tuple[str, list[str], Dict[str, str]]:
    """Validate STDIO executable allowlist, arguments, and isolated environment variables."""
    cmd = (command or "").strip()
    if not cmd:
        raise MCPSecurityError("STDIO command cannot be empty.")
    if _SHELL_META_RE.search(cmd):
        raise MCPSecurityError("STDIO command contains disallowed shell metacharacters.")

    exe_stem = Path(cmd).stem.lower()
    allowed = set(ALLOWED_STDIO_EXECUTABLES)
    if extra_allowlist:
        allowed.update(str(item).strip().lower() for item in extra_allowlist)
    if exe_stem not in allowed:
        raise MCPSecurityError(
            f"Executable '{Path(cmd).name}' is not in the STDIO allowlist: "
            f"{', '.join(sorted(allowed))}."
        )

    cleaned_args: list[str] = []
    for raw_arg in args or ():
        arg = str(raw_arg)
        if _SHELL_META_RE.search(arg):
            raise MCPSecurityError("STDIO argument contains disallowed shell metacharacters.")
        cleaned_args.append(arg)

    if allowed_roots:
        resolved_roots = [Path(r).resolve() for r in allowed_roots if str(r).strip()]
        for arg in cleaned_args:
            if (":" in arg and "\\" in arg) or arg.startswith(("/", "\\", "~")):
                candidate = Path(arg).expanduser().resolve()
                if not any(
                    candidate == root or root in candidate.parents
                    for root in resolved_roots
                ):
                    raise MCPSecurityError(
                        f"STDIO path argument '{arg}' is outside configured allowed_roots."
                    )

    # Build sanitized child environment: inherit minimal system vars + explicit env_refs only
    safe_env: Dict[str, str] = {}
    for key in ("PATH", "SYSTEMROOT", "COMSPEC", "TEMP", "TMP", "HOME", "USERPROFILE", "APPDATA"):
        if key in os.environ:
            safe_env[key] = os.environ[key]

    for env_key, env_source in (env_refs or {}).items():
        clean_key = str(env_key).strip()
        if not clean_key or clean_key.upper() in BLOCKED_PARENT_ENV_KEYS:
            continue
        source_name = str(env_source).strip()
        if source_name.startswith("env:"):
            var_name = source_name[4:].strip()
            if var_name in os.environ and var_name.upper() not in BLOCKED_PARENT_ENV_KEYS:
                safe_env[clean_key] = os.environ[var_name]
        elif source_name in os.environ and source_name.upper() not in BLOCKED_PARENT_ENV_KEYS:
            safe_env[clean_key] = os.environ[source_name]

    return cmd, cleaned_args, safe_env


def _is_disallowed_ip(ip_str: str) -> bool:
    addr = ipaddress.ip_address(ip_str)
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    )


def validate_remote_url(
    url: str,
    allow_insecure_http: bool = False,
    allow_private_network: bool = False,
    resolve_dns: bool = True,
) -> str:
    """Validate remote MCP server URL scheme and enforce SSRF destination protections."""
    raw = (url or "").strip()
    if not raw:
        raise MCPSecurityError("Remote MCP server URL cannot be empty.")

    parsed = urlparse(raw)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("https", "http"):
        raise MCPSecurityError(f"Unsupported URL scheme '{scheme}'. Only HTTPS/HTTP are supported.")
    if scheme == "http" and not allow_insecure_http:
        raise MCPSecurityError(
            "HTTPS is required for remote MCP servers unless allow_insecure_http is enabled."
        )
    if not parsed.hostname:
        raise MCPSecurityError("Remote MCP URL must include a valid hostname.")

    hostname = parsed.hostname.strip().lower()
    if not allow_private_network:
        if hostname in ("localhost", "localhost.localdomain") or hostname.endswith(".local"):
            raise MCPSecurityError(
                f"Blocked local hostname '{hostname}' due to SSRF protection."
            )
        # Check literal IP hostname first
        try:
            if _is_disallowed_ip(hostname):
                raise MCPSecurityError(
                    f"Blocked private/loopback IP '{hostname}' due to SSRF protection."
                )
        except ValueError:
            if resolve_dns:
                # Resolve DNS hostname and verify all destination IPs
                try:
                    infos = socket.getaddrinfo(hostname, parsed.port or (443 if scheme == "https" else 80))
                except socket.gaierror as exc:
                    raise MCPSecurityError(
                        f"Could not resolve remote MCP host '{hostname}': {exc}"
                    ) from exc
                for info in infos:
                    ip_val = info[4][0]
                    if _is_disallowed_ip(ip_val):
                        raise MCPSecurityError(
                            f"Remote MCP host '{hostname}' resolves to blocked internal IP '{ip_val}'."
                        )

    return raw


def build_stdio_spec_from_package(
    registry_type: str,
    identifier: str,
    version: str = "",
    runtime_hint: str = "",
    package_arguments: Optional[Sequence[Any]] = None,
) -> tuple[str, list[str], str]:
    """Build a strictly allowlisted (command, args, install_hint) tuple from registry package metadata.

    Never executes commands or allows shell metacharacters. Supports npm (`npx`),
    pypi (`uvx`/`python`), and oci/docker (`docker`).
    """
    pkg_type = (registry_type or "").strip().lower()
    pkg_id = (identifier or "").strip()
    if not pkg_id or _SHELL_META_RE.search(pkg_id):
        raise MCPSecurityError("Registry package identifier is empty or contains unsafe characters.")

    hint = (runtime_hint or "").strip().lower()
    extra_args: list[str] = []
    for item in package_arguments or ():
        if isinstance(item, Mapping):
            val = item.get("value") or item.get("name") or item.get("default")
            if val is not None and str(val).strip():
                extra_args.append(str(val).strip())
        elif item is not None and str(item).strip():
            extra_args.append(str(item).strip())

    target = f"{pkg_id}@{version.strip()}" if version.strip() and pkg_type == "npm" and "@" not in pkg_id[1:] else pkg_id
    if pkg_type == "npm" or hint in ("npx", "node"):
        cmd = "npx"
        args = ["-y", target, *extra_args]
        install_hint = f"npm / npx (`{pkg_id}`)"
    elif pkg_type == "pypi" or hint in ("uvx", "uv", "python", "python3"):
        if hint in ("python", "python3"):
            cmd = "python"
            args = ["-m", pkg_id.replace("-", "_"), *extra_args]
            install_hint = f"PyPI (`pip install {pkg_id}`)"
        else:
            cmd = "uvx"
            spec = f"{pkg_id}=={version.strip()}" if version.strip() else pkg_id
            args = [spec, *extra_args]
            install_hint = f"PyPI / uvx (`uvx {pkg_id}`)"
    elif pkg_type in ("oci", "docker") or hint == "docker":
        cmd = "docker"
        image = f"{pkg_id}:{version.strip()}" if version.strip() and ":" not in pkg_id else pkg_id
        args = ["run", "--rm", "-i", image, *extra_args]
        install_hint = f"OCI / Docker (`{image}`)"
    else:
        raise MCPSecurityError(
            f"Unsupported or unverified registry package type '{registry_type}' (runtimeHint='{runtime_hint}')."
        )

    validated_cmd, validated_args, _ = validate_stdio_config(command=cmd, args=args)
    return validated_cmd, validated_args, install_hint
