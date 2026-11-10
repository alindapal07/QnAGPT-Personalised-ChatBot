"""Normalized domain models for dynamic MCP Registry & Connector Marketplace discovery."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


_CATEGORY_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Developer Tools", ("git", "github", "gitlab", "code", "repo", "ide", "lint", "debug", "ci", "sentry", "linear")),
    ("Data & Databases", ("sql", "postgres", "sqlite", "mysql", "mongo", "redis", "database", "analytics", "bigquery", "snowflake")),
    ("Web & Search", ("search", "web", "fetch", "scrape", "crawl", "browser", "playwright", "puppeteer", "url")),
    ("Files & Storage", ("file", "filesystem", "drive", "s3", "storage", "box", "dropbox", "gcs")),
    ("Productivity", ("calendar", "email", "gmail", "notion", "asana", "jira", "trello", "task", "note", "docs")),
    ("Communication", ("slack", "discord", "teams", "chat", "message", "sms", "twilio")),
    ("Cloud & DevOps", ("aws", "gcp", "azure", "cloud", "kubernetes", "k8s", "docker", "terraform", "vercel", "cloudflare")),
    ("AI & Utilities", ("memory", "vector", "rag", "embed", "llm", "prompt", "time", "weather", "sequential")),
)


def infer_category(text: str) -> str:
    """Dynamically classify a registry listing from its metadata text & tool descriptions."""
    lowered = f" {(text or '').lower()} "
    best_cat = "General"
    best_score = 0
    for category, keywords in _CATEGORY_RULES:
        score = sum(1 for kw in keywords if kw in lowered)
        if score > best_score:
            best_score = score
            best_cat = category
    return best_cat


@dataclass(frozen=True)
class TransportOption:
    """Normalized connection option derived from a registry server's remotes or packages."""

    transport: str  # "streamable_http" | "sse" | "stdio"
    label: str
    url: str = ""
    command: str = ""
    args: Tuple[str, ...] = ()
    package_type: str = ""
    package_identifier: str = ""
    version: str = ""
    install_hint: str = ""
    required_headers: Tuple[str, ...] = ()
    env_variables: Tuple[Dict[str, Any], ...] = ()
    direct_connectable: bool = True
    notes: str = ""


@dataclass(frozen=True)
class RegistryListing:
    """Normalized MCP connector/server entry returned from any MCP directory provider."""

    connector_id: str
    registry_source: str  # "official_mcp" | "glama"
    registry_id: str
    name: str
    title: str
    description: str
    category: str = "General"
    categories: Tuple[str, ...] = ("General",)
    capabilities: Tuple[str, ...] = ()
    icon_url: str = ""
    listing_url: str = ""
    repository_url: str = ""
    homepage_url: str = ""
    license_name: str = ""
    attribution_name: str = ""
    attribution_url: str = ""
    transport_options: Tuple[TransportOption, ...] = ()
    auth_methods: Tuple[str, ...] = ("none",)
    env_variables: Tuple[Dict[str, Any], ...] = ()
    preview_tools: Tuple[Dict[str, Any], ...] = ()
    popularity_score: float = 0.0
    recommended: bool = False
    created_at: str = ""
    updated_at: str = ""
    version: str = ""
    tool_count: int = 0
    direct_connectable: bool = False
    connectability_reason: str = ""

    @property
    def primary_transport(self) -> str:
        if self.transport_options:
            return self.transport_options[0].transport
        return "manual"

    @property
    def primary_install_info(self) -> str:
        if self.transport_options:
            opt = self.transport_options[0]
            if opt.install_hint:
                return opt.install_hint
            if opt.url:
                return opt.url
        return "Manual repository setup"


@dataclass
class RegistryPage:
    """Paginated normalized response returned by a registry provider or aggregator."""

    items: List[RegistryListing]
    source: str
    next_cursor: Optional[str] = None
    has_next_page: bool = False
    total_count: Optional[int] = None
    from_cache: bool = False
    attribution_name: str = ""
    attribution_url: str = ""
    attributions: List[tuple[str, str]] = field(default_factory=list)
    warning: str = ""
