"""Glama MCP Directory (`https://glama.ai/api/mcp/v1`) provider with license & attribution compliance."""

from typing import Any, Dict, List, Mapping, Optional

from qna_bot.mcp.exceptions import MCPSecurityError
from qna_bot.mcp.registry.base import CachedRegistryHTTPClient, RegistryError
from qna_bot.mcp.registry.models import (
    RegistryListing,
    RegistryPage,
    TransportOption,
    infer_category,
)
from qna_bot.mcp.security import build_stdio_spec_from_package, validate_remote_url


class GlamaRegistryProvider:
    """Queries Glama's `/api/mcp/v1/servers` and `/v1/connectors` directories dynamically."""

    source_id = "glama"
    display_name = "Glama MCP Directory"
    attribution_url = "https://glama.ai"

    def __init__(
        self,
        http_client: CachedRegistryHTTPClient,
        base_url: str = "https://glama.ai/api/mcp/v1",
        api_key: str = "",
        allow_insecure_http: bool = False,
        allow_private_network: bool = False,
    ) -> None:
        self.http = http_client
        self.base_url = (base_url or "https://glama.ai/api/mcp/v1").rstrip("/")
        self.api_key = (api_key or "").strip()
        self.allow_insecure_http = allow_insecure_http
        self.allow_private_network = allow_private_network

    def _headers(self) -> Dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    def list_servers(
        self,
        query: str = "",
        limit: int = 24,
        cursor: Optional[str] = None,
        sort: str = "all",
    ) -> RegistryPage:
        # `/connectors` returns hosted 1-click connectable MCP integrations (`connection`, `toolCount`, `healthy`);
        # `/servers` covers newly indexed open-source/STDIO MCP repositories.
        endpoint = "/servers" if sort == "recent" else "/connectors"
        params: Dict[str, Any] = {
            "first": min(100, max(1, int(limit))),
        }
        if cursor:
            params["after"] = cursor
        if query.strip():
            params["query"] = query.strip()
        if endpoint == "/connectors":
            if sort == "recommended":
                params["sort"] = "recommended:desc"
            elif sort == "popular":
                params["sort"] = "popular:desc"

        if not self.api_key:
            return RegistryPage(
                items=[],
                source=self.source_id,
                attribution_name=self.display_name,
                attribution_url=self.attribution_url,
                warning=(
                    "Glama MCP Directory API requires a Bearer API key. "
                    "Set `GLAMA_API_KEY` in `.env` to browse hosted Glama connectors."
                ),
            )

        try:
            payload, from_cache = self.http.get_json(
                source=self.source_id,
                url=f"{self.base_url}{endpoint}",
                params=params,
                headers=self._headers(),
                allow_insecure_http=self.allow_insecure_http,
                allow_private_network=self.allow_private_network,
            )
        except RegistryError as exc:
            return RegistryPage(
                items=[],
                source=self.source_id,
                attribution_name=self.display_name,
                attribution_url=self.attribution_url,
                warning=str(exc),
            )

        raw_list = payload.get("connectors") or payload.get("servers") or payload.get("items") or []
        page_info = payload.get("pageInfo") if isinstance(payload.get("pageInfo"), Mapping) else {}
        listings: List[RegistryListing] = []
        for item in raw_list:
            norm = self._normalize_item(item, is_connector=(endpoint == "/connectors"))
            if norm is not None:
                listings.append(norm)

        has_next = bool(page_info.get("hasNextPage"))
        end_cursor = page_info.get("endCursor")
        return RegistryPage(
            items=listings,
            source=self.source_id,
            next_cursor=str(end_cursor) if end_cursor and has_next else None,
            has_next_page=has_next,
            from_cache=from_cache,
            attribution_name=self.display_name,
            attribution_url=self.attribution_url,
        )

    def get_server_details(self, registry_id: str) -> Optional[RegistryListing]:
        if not self.api_key:
            return None
        clean_id = registry_id.strip().lstrip("/")
        payload, _ = self.http.get_json(
            source=self.source_id,
            url=f"{self.base_url}/servers/{clean_id}",
            headers=self._headers(),
            allow_insecure_http=self.allow_insecure_http,
            allow_private_network=self.allow_private_network,
        )
        if not payload:
            return None
        return self._normalize_item(payload, is_connector=False)

    def _normalize_item(self, item: Any, is_connector: bool = False) -> Optional[RegistryListing]:
        if not isinstance(item, Mapping):
            return None

        namespace = str(item.get("namespace") or "").strip()
        slug = str(item.get("slug") or "").strip()
        raw_id = str(item.get("id") or "").strip()
        full_slug = f"{namespace}/{slug}".strip("/") if (namespace or slug) else (raw_id or str(item.get("name") or ""))
        if not full_slug:
            return None

        title = str(
            item.get("displayName") or item.get("name") or slug or full_slug
        ).strip()
        description = str(item.get("description") or "").strip()

        # Glama requires linking directly to the server's Glama listing URL
        listing_url = str(item.get("url") or "").strip()
        if not listing_url:
            kind_path = "connectors" if is_connector else "servers"
            listing_url = f"https://glama.ai/mcp/{kind_path}/{full_slug}"

        repo_obj = item.get("repository") if isinstance(item.get("repository"), Mapping) else {}
        repo_url = str(repo_obj.get("url") or "").strip()
        spdx = item.get("spdxLicense") if isinstance(item.get("spdxLicense"), Mapping) else {}
        license_name = str(spdx.get("name") or "").strip()

        # Extract environment variable JSON Schema
        envs: List[Dict[str, Any]] = []
        env_schema = item.get("environmentVariablesJsonSchema")
        if isinstance(env_schema, Mapping):
            props = env_schema.get("properties") if isinstance(env_schema.get("properties"), Mapping) else {}
            req_set = set(env_schema.get("required") or [])
            for k, meta in props.items():
                meta_map = meta if isinstance(meta, Mapping) else {}
                envs.append(
                    {
                        "name": str(k),
                        "description": str(meta_map.get("description") or ""),
                        "required": str(k) in req_set,
                        "isSecret": True,
                    }
                )

        # Extract tools preview dynamically reported by Glama
        raw_tools = item.get("tools") or []
        preview_tools: List[Dict[str, Any]] = []
        if isinstance(raw_tools, list):
            for t in raw_tools:
                if isinstance(t, Mapping) and t.get("name"):
                    preview_tools.append(
                        {
                            "name": str(t["name"]),
                            "description": str(t.get("description") or ""),
                            "inputSchema": t.get("inputSchema") or t.get("input_schema") or {},
                        }
                    )

        attrs = item.get("attributes")
        attr_tags: List[str] = []
        if isinstance(attrs, list):
            attr_tags = [str(a) for a in attrs if a]
        elif isinstance(attrs, Mapping):
            attr_tags = [f"{k}:{v}" for k, v in attrs.items()]

        transports: List[TransportOption] = []
        auth_methods: List[str] = []

        # Parse Glama live `/connectors` `connection` block as well as legacy endpointUrl/remoteUrl fields
        conn_obj = item.get("connection") if isinstance(item.get("connection"), Mapping) else {}
        remote_url = str(
            conn_obj.get("url")
            or item.get("endpointUrl")
            or item.get("mcpUrl")
            or item.get("remoteUrl")
            or ""
        ).strip()
        raw_conn_tr = str(conn_obj.get("transport") or "streamable_http").strip().lower()
        norm_conn_tr = "sse" if raw_conn_tr == "sse" else "streamable_http"
        raw_conn_auth = str(conn_obj.get("authType") or "").strip().lower()

        if remote_url:
            valid_remote = False
            notes = ""
            try:
                validate_remote_url(remote_url, allow_insecure_http=False, allow_private_network=False, resolve_dns=False)
                valid_remote = True
            except MCPSecurityError as exc:
                notes = str(exc)
            transports.append(
                TransportOption(
                    transport=norm_conn_tr,
                    label=f"Remote ({norm_conn_tr})",
                    url=remote_url,
                    package_type="remote",
                    install_hint=remote_url,
                    direct_connectable=valid_remote,
                    notes=notes,
                )
            )
            if raw_conn_auth in ("oauth", "oauth2") or any("oauth" in t.lower() for t in attr_tags):
                auth_methods.append("oauth")
            elif raw_conn_auth == "none" or "auth:none" in attr_tags:
                auth_methods.append("none")
            elif "api" in raw_conn_auth:
                auth_methods.append("api_key")
            else:
                auth_methods.append("bearer_token")

        for pkg in item.get("packages") or ():
            if not isinstance(pkg, Mapping):
                continue
            try:
                cmd, args, hint_str = build_stdio_spec_from_package(
                    registry_type=str(pkg.get("registryType") or pkg.get("type") or ""),
                    identifier=str(pkg.get("identifier") or pkg.get("name") or ""),
                    version=str(pkg.get("version") or ""),
                    runtime_hint=str(pkg.get("runtimeHint") or ""),
                )
                transports.append(
                    TransportOption(
                        transport="stdio",
                        label=f"Local STDIO ({cmd})",
                        command=cmd,
                        args=tuple(args),
                        package_type=str(pkg.get("registryType") or cmd).lower(),
                        package_identifier=str(pkg.get("identifier") or pkg.get("name") or ""),
                        install_hint=hint_str,
                        env_variables=tuple(envs),
                        direct_connectable=True,
                        notes="Requires explicit user confirmation before launching local process.",
                    )
                )
            except MCPSecurityError:
                continue

        if not auth_methods:
            if any("oauth" in t.lower() for t in attr_tags):
                auth_methods = ["oauth"]
            else:
                auth_methods = ["api_key"] if envs else ["none"]

        connectable = any(t.direct_connectable for t in transports)
        reason = "" if connectable else (
            "Listing references source repository or requires external deployment before direct connection."
        )
        glama_cat_map = {
            "search": "Web & Search",
            "developer-tools": "Developer Tools",
            "databases": "Data & Databases",
            "data": "Data & Databases",
            "productivity": "Productivity",
            "communication": "Communication",
            "storage": "Files & Storage",
            "cloud": "Cloud & DevOps",
        }
        explicit_cats = [
            glama_cat_map.get(t.split(":", 1)[1], t.split(":", 1)[1].replace("-", " ").title())
            for t in attr_tags
            if t.startswith("category:") and ":" in t
        ]
        cat_corpus = " ".join([title, description, *attr_tags, *(t["name"] for t in preview_tools)])
        category = (
            next((c for c in explicit_cats if c in glama_cat_map.values()), None)
            or infer_category(cat_corpus)
        )
        all_categories = tuple(dict.fromkeys([category, *explicit_cats]))

        raw_tool_count = item.get("toolCount")
        try:
            tool_count_val = max(len(preview_tools), int(raw_tool_count or 0))
        except (TypeError, ValueError):
            tool_count_val = len(preview_tools)

        q_score = float(item.get("qualityScore") or 0.0)
        healthy_bonus = 1.0 if item.get("healthy") is True else 0.0
        pop_score = q_score + healthy_bonus + (2.0 if is_connector else (1.2 if connectable else 0.5))

        return RegistryListing(
            connector_id=f"glama::{full_slug}",
            registry_source=self.source_id,
            registry_id=full_slug,
            name=full_slug,
            title=title,
            description=description,
            category=category,
            categories=all_categories,
            capabilities=tuple(t["name"] for t in preview_tools[:12]) or tuple(attr_tags[:8]),
            icon_url=str(item.get("thumbnailUrl") or item.get("iconUrl") or item.get("logoUrl") or "").strip(),
            listing_url=listing_url,
            repository_url=repo_url,
            homepage_url=listing_url,
            license_name=license_name,
            attribution_name=self.display_name,
            attribution_url=self.attribution_url,
            transport_options=tuple(transports),
            auth_methods=tuple(dict.fromkeys(auth_methods)),
            env_variables=tuple(envs),
            preview_tools=tuple(preview_tools),
            popularity_score=pop_score,
            recommended=bool(item.get("healthy")) or is_connector or any("recommended" in a.lower() for a in attr_tags),
            created_at=str(item.get("createdAt") or ""),
            updated_at=str(item.get("updatedAt") or ""),
            tool_count=tool_count_val,
            direct_connectable=connectable,
            connectability_reason=reason,
        )
