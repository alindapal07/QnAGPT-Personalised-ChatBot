"""Official Model Context Protocol Registry (`registry.modelcontextprotocol.io`) provider."""

from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import quote

from qna_bot.mcp.exceptions import MCPSecurityError
from qna_bot.mcp.registry.base import CachedRegistryHTTPClient
from qna_bot.mcp.registry.models import (
    RegistryListing,
    RegistryPage,
    TransportOption,
    infer_category,
)
from qna_bot.mcp.security import build_stdio_spec_from_package, validate_remote_url


class OfficialMCPRegistryProvider:
    """Fetches and normalizes MCP servers from the Official MCP Registry REST API (`v0.1`)."""

    source_id = "official_mcp"
    display_name = "Official MCP Registry"
    attribution_url = "https://registry.modelcontextprotocol.io"

    def __init__(
        self,
        http_client: CachedRegistryHTTPClient,
        base_url: str = "https://registry.modelcontextprotocol.io",
        allow_insecure_http: bool = False,
        allow_private_network: bool = False,
    ) -> None:
        self.http = http_client
        self.base_url = (base_url or "https://registry.modelcontextprotocol.io").rstrip("/")
        self.allow_insecure_http = allow_insecure_http
        self.allow_private_network = allow_private_network

    def list_servers(
        self,
        query: str = "",
        limit: int = 24,
        cursor: Optional[str] = None,
        sort: str = "all",
    ) -> RegistryPage:
        params: Dict[str, Any] = {
            "limit": min(100, max(1, int(limit))),
        }
        if cursor:
            params["cursor"] = cursor
        if query.strip():
            params["search"] = query.strip()

        payload, from_cache = self.http.get_json(
            source=self.source_id,
            url=f"{self.base_url}/v0.1/servers",
            params=params,
            allow_insecure_http=self.allow_insecure_http,
            allow_private_network=self.allow_private_network,
        )
        raw_servers = payload.get("servers") or []
        metadata = payload.get("metadata") or {}
        listings: List[RegistryListing] = []
        seen_names: set[str] = set()
        for entry in raw_servers:
            listing = self._normalize_entry(entry)
            if listing is None or listing.registry_id in seen_names:
                continue
            seen_names.add(listing.registry_id)
            listings.append(listing)

        next_cursor = metadata.get("nextCursor")
        if sort == "recent":
            listings.sort(key=lambda x: x.updated_at or x.created_at or "", reverse=True)
        elif sort == "popular":
            listings.sort(key=lambda x: (x.popularity_score, x.title.lower()), reverse=True)

        return RegistryPage(
            items=listings,
            source=self.source_id,
            next_cursor=str(next_cursor) if next_cursor else None,
            has_next_page=bool(next_cursor),
            total_count=metadata.get("count"),
            from_cache=from_cache,
            attribution_name=self.display_name,
            attribution_url=self.attribution_url,
        )

    def get_server_details(self, registry_id: str) -> Optional[RegistryListing]:
        encoded = quote(registry_id.strip(), safe="")
        payload, _ = self.http.get_json(
            source=self.source_id,
            url=f"{self.base_url}/v0.1/servers/{encoded}/versions/latest",
            allow_insecure_http=self.allow_insecure_http,
            allow_private_network=self.allow_private_network,
        )
        if not payload:
            return None
        if "server" in payload:
            return self._normalize_entry(payload)
        return self._normalize_entry({"server": payload})

    def _normalize_entry(self, entry: Any) -> Optional[RegistryListing]:
        if not isinstance(entry, Mapping):
            return None
        srv = entry.get("server") if isinstance(entry.get("server"), Mapping) else entry
        meta = entry.get("_meta") if isinstance(entry.get("_meta"), Mapping) else {}
        official_meta = meta.get("io.modelcontextprotocol.registry/official")
        if not isinstance(official_meta, Mapping):
            official_meta = meta

        name = str(srv.get("name") or "").strip()
        if not name:
            return None
        title = str(srv.get("title") or "").strip() or name.split("/")[-1].replace("-", " ").title()
        description = str(srv.get("description") or "").strip()
        version = str(srv.get("version") or "").strip()
        repo = srv.get("repository") if isinstance(srv.get("repository"), Mapping) else {}
        repo_url = str(repo.get("url") or "").strip()
        homepage = str(srv.get("websiteUrl") or repo_url or "").strip()

        icons = srv.get("icons") or []
        icon_url = ""
        if isinstance(icons, list) and icons and isinstance(icons[0], Mapping):
            icon_url = str(icons[0].get("src") or icons[0].get("url") or "").strip()

        transports: List[TransportOption] = []
        auth_methods: List[str] = []
        collected_envs: List[Dict[str, Any]] = []

        # 1. Remote transports (`streamable-http`, `sse`)
        for remote in srv.get("remotes") or ():
            if not isinstance(remote, Mapping):
                continue
            raw_type = str(remote.get("type") or remote.get("transport") or "").strip().lower()
            norm_transport = "streamable_http" if raw_type in ("streamable-http", "streamable_http", "http") else (
                "sse" if raw_type == "sse" else ""
            )
            url = str(remote.get("url") or "").strip()
            if not norm_transport or not url:
                continue

            req_headers: List[str] = []
            raw_headers = remote.get("headers")
            if isinstance(raw_headers, list):
                for h in raw_headers:
                    if isinstance(h, Mapping):
                        h_name = str(h.get("name") or "").strip()
                        if h_name:
                            req_headers.append(h_name)
                            lower_h = h_name.lower()
                            if "authorization" in lower_h:
                                auth_methods.append("bearer_token")
                            elif "key" in lower_h or "token" in lower_h:
                                auth_methods.append("api_key")
            elif isinstance(raw_headers, Mapping):
                for h_name, h_val in raw_headers.items():
                    req_headers.append(str(h_name))
                    if "bearer" in str(h_val).lower() or "authorization" in str(h_name).lower():
                        auth_methods.append("bearer_token")
                    else:
                        auth_methods.append("api_key")

            is_valid_remote = False
            notes = ""
            try:
                validate_remote_url(url, allow_insecure_http=False, allow_private_network=False, resolve_dns=False)
                is_valid_remote = True
            except MCPSecurityError as exc:
                notes = str(exc)

            transports.append(
                TransportOption(
                    transport=norm_transport,
                    label=f"Remote ({norm_transport})",
                    url=url,
                    package_type="remote",
                    install_hint=url,
                    required_headers=tuple(req_headers),
                    direct_connectable=is_valid_remote,
                    notes=notes,
                )
            )
            if not req_headers and not auth_methods:
                auth_methods.append("oauth")

        # 2. Local STDIO packages (`npm`, `pypi`, `oci`/`docker`)
        for pkg in srv.get("packages") or ():
            if not isinstance(pkg, Mapping):
                continue
            reg_type = str(pkg.get("registryType") or pkg.get("type") or "").strip()
            ident = str(pkg.get("identifier") or pkg.get("name") or "").strip()
            pkg_ver = str(pkg.get("version") or "").strip()
            hint = str(pkg.get("runtimeHint") or "").strip()
            pkg_args = pkg.get("packageArguments") or pkg.get("runtimeArguments") or ()

            pkg_envs: List[Dict[str, Any]] = []
            raw_envs = pkg.get("environmentVariables") or []
            if isinstance(raw_envs, list):
                for ev in raw_envs:
                    if isinstance(ev, Mapping) and ev.get("name"):
                        ev_item = {
                            "name": str(ev["name"]).strip(),
                            "description": str(ev.get("description") or "").strip(),
                            "required": bool(ev.get("isRequired", ev.get("required", False))),
                            "isSecret": bool(ev.get("isSecret", True)),
                        }
                        pkg_envs.append(ev_item)
                        collected_envs.append(ev_item)

            try:
                cmd, args, install_hint = build_stdio_spec_from_package(
                    registry_type=reg_type,
                    identifier=ident,
                    version=pkg_ver,
                    runtime_hint=hint,
                    package_arguments=pkg_args if isinstance(pkg_args, (list, tuple)) else (),
                )
                transports.append(
                    TransportOption(
                        transport="stdio",
                        label=f"Local STDIO ({reg_type or cmd})",
                        command=cmd,
                        args=tuple(args),
                        package_type=reg_type.lower() or cmd,
                        package_identifier=ident,
                        version=pkg_ver,
                        install_hint=install_hint,
                        env_variables=tuple(pkg_envs),
                        direct_connectable=True,
                        notes="Requires explicit user confirmation before launching local process.",
                    )
                )
            except MCPSecurityError as exc:
                transports.append(
                    TransportOption(
                        transport="stdio",
                        label=f"Package ({reg_type or 'custom'})",
                        package_type=reg_type.lower(),
                        package_identifier=ident,
                        version=pkg_ver,
                        install_hint=f"{reg_type}: {ident}".strip(": "),
                        env_variables=tuple(pkg_envs),
                        direct_connectable=False,
                        notes=str(exc),
                    )
                )

        if not auth_methods:
            auth_methods = ["none"] if not collected_envs else ["api_key"]

        connectable = any(t.direct_connectable for t in transports)
        reason = ""
        if not connectable:
            reason = (
                transports[0].notes
                if transports and transports[0].notes
                else "No runnable STDIO package or remote HTTP/SSE endpoint declared in registry metadata."
            )

        short_slug = name.split("/")[-1]
        category = infer_category(f"{short_slug} {title} {description}")
        created_at = str(official_meta.get("publishedAt") or official_meta.get("created_at") or "")
        updated_at = str(official_meta.get("updatedAt") or official_meta.get("updated_at") or created_at)
        listing_url = f"https://registry.modelcontextprotocol.io/servers/{quote(name, safe='/')}"

        return RegistryListing(
            connector_id=f"official_mcp::{name}",
            registry_source=self.source_id,
            registry_id=name,
            name=name,
            title=title,
            description=description,
            category=category,
            categories=(category,),
            capabilities=tuple(t.transport for t in transports),
            icon_url=icon_url,
            listing_url=listing_url,
            repository_url=repo_url,
            homepage_url=homepage,
            license_name="",
            attribution_name=self.display_name,
            attribution_url=self.attribution_url,
            transport_options=tuple(transports),
            auth_methods=tuple(dict.fromkeys(auth_methods)),
            env_variables=tuple(collected_envs),
            popularity_score=1.0 if connectable else 0.3,
            recommended=connectable and any(t.transport == "streamable_http" for t in transports),
            created_at=created_at,
            updated_at=updated_at,
            version=version,
            direct_connectable=connectable,
            connectability_reason=reason,
        )
