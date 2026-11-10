"""Aggregated MCP Marketplace catalog service across Official MCP Registry and Glama."""

from typing import List, Optional, Sequence

from qna_bot.config.settings import AppSettings, get_settings
from qna_bot.database.connection import DatabaseManager
from qna_bot.database.repositories import RegistryCacheRepository
from qna_bot.mcp.exceptions import MCPSecurityError
from qna_bot.mcp.registry.base import CachedRegistryHTTPClient, RegistryError
from qna_bot.mcp.registry.glama_provider import GlamaRegistryProvider
from qna_bot.mcp.registry.models import RegistryListing, RegistryPage
from qna_bot.mcp.registry.official_provider import OfficialMCPRegistryProvider


class MCPRegistryService:
    """Coordinates dynamic server discovery, caching, search, filtering, and provider attribution."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        settings: Optional[AppSettings] = None,
        official_provider: Optional[OfficialMCPRegistryProvider] = None,
        glama_provider: Optional[GlamaRegistryProvider] = None,
        http_client: Optional[CachedRegistryHTTPClient] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.db = db_manager or DatabaseManager()
        self.cache_repo = RegistryCacheRepository(self.db)
        self.http = http_client or CachedRegistryHTTPClient(
            cache_repo=self.cache_repo,
            cache_ttl_sec=self.settings.registry_cache_ttl_sec,
        )
        self.official = official_provider or OfficialMCPRegistryProvider(
            http_client=self.http,
            base_url=self.settings.official_registry_url,
        )
        self.glama = glama_provider or GlamaRegistryProvider(
            http_client=self.http,
            base_url=self.settings.glama_api_url,
            api_key=self.settings.glama_api_key,
        )

    def clear_cache(self, source: Optional[str] = None) -> None:
        self.cache_repo.clear(source=source)

    @staticmethod
    def _matches_filters(
        item: RegistryListing,
        query: str,
        category: str,
        transport_filter: str,
        only_connectable: bool,
    ) -> bool:
        if category and category != "All":
            if item.category != category and category not in item.categories:
                return False

        if transport_filter and transport_filter != "All":
            if not any(t.transport == transport_filter for t in item.transport_options):
                return False

        if only_connectable and not item.direct_connectable:
            return False

        if query.strip():
            q = query.strip().lower()
            haystack = " ".join(
                [
                    item.name,
                    item.title,
                    item.description,
                    item.category,
                    " ".join(item.capabilities),
                    " ".join(str(t.get("name", "")) for t in item.preview_tools),
                ]
            ).lower()
            if q not in haystack:
                return False

        return True

    def discover(
        self,
        source: str = "all",
        query: str = "",
        category: str = "All",
        feed: str = "all",
        transport_filter: str = "All",
        only_connectable: bool = False,
        limit: int = 24,
        cursor: Optional[str] = None,
    ) -> RegistryPage:
        """Query selected registry provider(s), filter cleanly, and return normalized page + attribution."""
        warnings: List[str] = []
        items: List[RegistryListing] = []
        next_cursor: Optional[str] = None
        has_next = False
        from_cache = False
        attr_names: List[str] = []
        attr_urls: List[str] = []

        attributions: list[tuple[str, str]] = []
        providers = []
        if source in ("all", "official_mcp"):
            providers.append(self.official)
        if source in ("all", "glama"):
            providers.append(self.glama)

        for provider in providers:
            try:
                page = provider.list_servers(
                    query=query,
                    limit=limit,
                    cursor=cursor if source != "all" else None,
                    sort=feed,
                )
                items.extend(page.items)
                if page.attribution_name and page.attribution_url:
                    pair = (page.attribution_name, page.attribution_url)
                    if pair not in attributions:
                        attributions.append(pair)
                if page.attribution_name:
                    attr_names.append(page.attribution_name)
                    attr_urls.append(page.attribution_url)
                if page.warning:
                    warnings.append(page.warning)
                from_cache = from_cache or page.from_cache
                if source != "all":
                    next_cursor = page.next_cursor
                    has_next = page.has_next_page
            except (RegistryError, MCPSecurityError) as exc:
                warnings.append(f"{provider.display_name}: {exc}")

        filtered = [
            it
            for it in items
            if self._matches_filters(
                it,
                query=query,
                category=category,
                transport_filter=transport_filter,
                only_connectable=only_connectable,
            )
        ]

        if feed == "recommended":
            filtered.sort(key=lambda x: (x.recommended, x.direct_connectable, x.popularity_score), reverse=True)
        elif feed == "popular":
            filtered.sort(key=lambda x: (x.popularity_score, x.direct_connectable, len(x.preview_tools)), reverse=True)
        elif feed == "recent":
            filtered.sort(key=lambda x: (x.updated_at or x.created_at or ""), reverse=True)

        return RegistryPage(
            items=filtered,
            source=source,
            next_cursor=next_cursor,
            has_next_page=has_next,
            total_count=len(filtered),
            from_cache=from_cache,
            attribution_name=" & ".join(dict.fromkeys(attr_names)) or "Official MCP Registry",
            attribution_url=attr_urls[0] if attr_urls else self.official.attribution_url,
            attributions=attributions or [(self.official.display_name, self.official.attribution_url)],
            warning=" | ".join(warnings),
        )

    def list_categories(self, items: Sequence[RegistryListing]) -> List[str]:
        cats = sorted({item.category for item in items if item.category})
        return ["All", *cats]

    def get_details(self, registry_source: str, registry_id: str) -> Optional[RegistryListing]:
        if registry_source == "glama":
            return self.glama.get_server_details(registry_id)
        if registry_source == "official_mcp":
            return self.official.get_server_details(registry_id)
        return None
