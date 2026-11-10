"""DuckDuckGo web search service, SSRF-safe webpage extractor, and LangChain browser tools."""

from dataclasses import dataclass, field
from html import unescape
import json
import re
import threading
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from ddgs import DDGS
from ddgs.exceptions import DDGSException
import httpx
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from qna_bot.core.logging_config import get_logger
from qna_bot.mcp.exceptions import MCPSecurityError
from qna_bot.mcp.security import sanitize_tool_output, validate_remote_url

logger = get_logger(__name__)

_SAFE_SEARCH_BACKENDS = ("yahoo", "duckduckgo", "brave", "startpage")
_MAX_HTML_BYTES = 1024 * 1024
_MAX_EXTRACTED_CHARS = 18000
_DEFAULT_PAGE_SLICE_CHARS = 5500
_MAX_REDIRECT_HOPS = 3
_NOISE_ELEMENT_MARKERS = (
    "cookie",
    "consent",
    "newsletter",
    "sidebar",
    "advert",
    "promo",
    "social-share",
    "breadcrumb",
    "navbar",
    "menu",
    "footer",
)


@dataclass(frozen=True)
class WebSearchResult:
    """Structured web search hit preserving source URL."""

    title: str
    url: str
    snippet: str

    def to_dict(self) -> Dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


@dataclass(frozen=True)
class ExtractedWebPage:
    """Structured result from fetching and extracting readable text from a webpage."""

    requested_url: str
    final_url: str
    canonical_url: str
    title: str
    description: str
    content: str
    tables: List[str] = field(default_factory=list)
    status_code: int = 0
    success: bool = False
    error: str = ""
    char_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requested_url": self.requested_url,
            "final_url": self.final_url,
            "canonical_url": self.canonical_url,
            "title": self.title,
            "description": self.description,
            "content": self.content,
            "tables": list(self.tables),
            "status_code": self.status_code,
            "success": self.success,
            "error": self.error,
            "char_count": self.char_count,
        }


class WebSearchInput(BaseModel):
    """Typed input schema for the DuckDuckGo web search tool."""

    query: str = Field(..., min_length=1, description="Concise web search query.")


class OpenWebPageInput(BaseModel):
    """Typed input schema for reading a web search result or HTTPS URL."""

    url: str = Field(
        default="",
        description="Full https:// URL to open. Leave empty if supplying result_id.",
    )
    result_id: Optional[int] = Field(
        default=None,
        ge=0,
        le=20,
        description="Search result index returned by web_search (e.g., 1 for [1]). Never confuse with cursor.",
    )
    cursor: int = Field(
        default=0,
        ge=0,
        description="Character offset within the extracted webpage content for reading further sections.",
    )


def _table_to_markdown(table_tag: Any) -> str:
    """Convert an HTML <table> tag into a compact Markdown table."""
    rows: List[List[str]] = []
    for tr in table_tag.find_all("tr"):
        cells = [
            re.sub(r"\s+", " ", unescape(cell.get_text(" ", strip=True))).replace("|", "/").strip()
            for cell in tr.find_all(["th", "td"])
        ]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""

    col_count = max(len(r) for r in rows)
    normalized = [r + [""] * (col_count - len(r)) for r in rows]
    lines = [
        "| " + " | ".join(normalized[0]) + " |",
        "| " + " | ".join(["---"] * col_count) + " |",
    ]
    for row in normalized[1:40]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _extract_embedded_json_text(soup: BeautifulSoup) -> str:
    """Fallback extraction of readable JSON-LD or Next.js hydration payloads for JS-heavy pages."""
    chunks: List[str] = []
    for script in soup.find_all("script"):
        script_type = str(script.get("type") or "").lower()
        script_id = str(script.get("id") or "")
        raw = (script.string or script.get_text() or "").strip()
        if not raw or len(raw) > 120_000:
            continue
        if script_type == "application/ld+json" or script_id == "__NEXT_DATA__":
            try:
                parsed = json.loads(raw)
                pretty = json.dumps(parsed, ensure_ascii=False)
                # Keep readable prose/fields up to bounded limit
                chunks.append(pretty[:4000])
            except Exception:
                continue
    return "\n".join(chunks[:2]).strip()


def extract_readable_html(raw_html: str, base_url: str = "") -> tuple[str, str, str, str, List[str]]:
    """Extract (title, description, canonical_url, normalized_markdown, tables_md) cleanly from HTML."""
    parser_name = "lxml"
    try:
        soup = BeautifulSoup(raw_html, parser_name)
    except Exception:
        soup = BeautifulSoup(raw_html, "html.parser")

    title = ""
    if soup.title and soup.title.string:
        title = re.sub(r"\s+", " ", unescape(soup.title.string)).strip()
    if not title:
        og_title = soup.find("meta", attrs={"property": "og:title"})
        if og_title and og_title.get("content"):
            title = str(og_title["content"]).strip()

    description = ""
    desc_tag = soup.find("meta", attrs={"name": "description"}) or soup.find(
        "meta", attrs={"property": "og:description"}
    )
    if desc_tag and desc_tag.get("content"):
        description = re.sub(r"\s+", " ", unescape(str(desc_tag["content"]))).strip()

    canonical_url = base_url
    canon_tag = soup.find("link", attrs={"rel": lambda v: v and "canonical" in v})
    if canon_tag and canon_tag.get("href"):
        canonical_url = urljoin(base_url, str(canon_tag["href"]).strip()) or base_url

    embedded_fallback = _extract_embedded_json_text(soup)

    for tag_name in ("script", "style", "noscript", "svg", "iframe", "nav", "header", "footer", "aside", "form"):
        for node in soup.find_all(tag_name):
            node.decompose()

    for elem in list(soup.find_all(True)):
        if getattr(elem, "attrs", None) is None:
            continue
        classes = " ".join(elem.get("class") or [])
        elem_id = str(elem.get("id") or "")
        marker_text = f"{classes} {elem_id}".lower()
        if any(m in marker_text for m in _NOISE_ELEMENT_MARKERS):
            # Avoid removing the primary article/main wrapper itself if it matches a broad keyword
            if elem.name not in ("html", "body", "main", "article"):
                elem.decompose()

    candidates = [c for c in (soup.find("article"), soup.find("main")) if c is not None]
    root = None
    for cand in candidates:
        if len(cand.get_text(" ", strip=True)) >= 200:
            root = cand
            break
    if root is None:
        root = soup.body or soup

    tables_md: List[str] = []
    blocks: List[str] = []

    for table in root.find_all("table"):
        md_table = _table_to_markdown(table)
        if md_table:
            tables_md.append(md_table)
            table.replace_with(soup.new_string(f"\n\n{md_table}\n\n"))
        else:
            table.decompose()

    seen_lines: set[str] = set()
    for elem in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "blockquote", "pre"]):
        text = re.sub(r"\s+", " ", unescape(elem.get_text(" ", strip=True))).strip()
        if not text or len(text) < 3 or text in seen_lines:
            continue
        seen_lines.add(text)
        if elem.name == "h1":
            blocks.append(f"# {text}")
        elif elem.name == "h2":
            blocks.append(f"## {text}")
        elif elem.name in ("h3", "h4"):
            blocks.append(f"### {text}")
        elif elem.name == "li":
            blocks.append(f"- {text}")
        else:
            blocks.append(text)

    for tbl in tables_md:
        if tbl not in "\n\n".join(blocks):
            blocks.append(tbl)

    body = "\n\n".join(blocks).strip()
    if len(body) < 120:
        fallback_plain = re.sub(r"\s+", " ", unescape(root.get_text(" ", strip=True))).strip()
        if len(fallback_plain) > len(body):
            body = fallback_plain

    if len(body) < 200 and embedded_fallback:
        body = f"{body}\n\n[Structured Hydration Data]\n{embedded_fallback}".strip()

    body = re.sub(r"\n{3,}", "\n\n", body)[:_MAX_EXTRACTED_CHARS]
    return title, description, canonical_url, body, tables_md


class DuckDuckGoWebSearcher:
    """Wrapper around DuckDuckGo/DDGS search returning structured results with resilient backend fallback."""

    def __init__(self, max_results: int = 5) -> None:
        self.max_results = max_results
        self._lock = threading.RLock()
        self._recent_results: List[WebSearchResult] = []

    def _record_results(self, results: List[WebSearchResult]) -> None:
        with self._lock:
            if results:
                self._recent_results = list(results)

    def get_result_by_id(self, result_id: int) -> Optional[WebSearchResult]:
        """Look up a 1-based result (also tolerating 0 -> first item) from recent search hits."""
        with self._lock:
            idx = 0 if int(result_id) <= 1 else int(result_id) - 1
            if 0 <= idx < len(self._recent_results):
                return self._recent_results[idx]
            return None

    def get_recent_results(self) -> List[WebSearchResult]:
        with self._lock:
            return list(self._recent_results)

    @staticmethod
    def _normalize_items(raw_items: List[Dict[str, Any]]) -> List[WebSearchResult]:
        results: List[WebSearchResult] = []
        seen_urls: set[str] = set()
        for item in raw_items or []:
            title = str(item.get("title") or "Web Result").strip()
            url = str(item.get("link") or item.get("url") or item.get("href") or "").strip()
            snippet = str(item.get("snippet") or item.get("body") or "").strip()
            if url and url in seen_urls:
                continue
            if snippet or url:
                if url:
                    seen_urls.add(url)
                results.append(WebSearchResult(title=title, url=url, snippet=snippet))
        return results

    def _search_ddgs_fallback(self, cleaned_query: str, limit: int) -> List[WebSearchResult]:
        for backend in _SAFE_SEARCH_BACKENDS:
            try:
                raw = DDGS(timeout=6).text(
                    cleaned_query,
                    region="us-en",
                    max_results=limit,
                    backend=backend,
                )
                normalized = self._normalize_items(raw or [])
                if normalized:
                    return normalized
            except Exception as exc:
                logger.debug("DDGS fallback backend '%s' failed: %s", backend, exc)
        return []

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
    ) -> tuple[List[WebSearchResult], str]:
        """Run a web search and return (results, error_message)."""
        cleaned = (query or "").strip()
        if not cleaned:
            return [], "Empty web search query."

        limit = max_results or self.max_results
        try:
            wrapper = DuckDuckGoSearchAPIWrapper(max_results=limit)
            is_real_wrapper = type(wrapper).__module__.startswith("langchain_community")
            if is_real_wrapper:
                # Default region='wt-wt' triggers wt.wikipedia.org DNS failures inside ddgs.
                wrapper.region = "us-en"
                wrapper.backend = ",".join(_SAFE_SEARCH_BACKENDS)

            raw_items: List[Dict[str, Any]] = []
            ddgs_error: Optional[Exception] = None
            try:
                raw_items = wrapper.results(cleaned, max_results=limit)
            except DDGSException as exc:
                ddgs_error = exc
                logger.info("Primary DDGS search raised %s; trying resilient backend fallback.", exc)

            results = self._normalize_items(raw_items)
            if not results and is_real_wrapper:
                results = self._search_ddgs_fallback(cleaned, limit=limit)

            if results:
                self._record_results(results)
                return results, ""

            if ddgs_error is not None and not is_real_wrapper:
                raise ddgs_error

            if not is_real_wrapper:
                runner = DuckDuckGoSearchRun(api_wrapper=wrapper)
                text_out = str(runner.invoke(cleaned) or "").strip()
                if text_out and "No good DuckDuckGo Search Result" not in text_out:
                    hits = [WebSearchResult(title="DuckDuckGo Search", url="", snippet=text_out)]
                    self._record_results(hits)
                    return hits, ""
            return [], ""
        except Exception as exc:
            logger.warning("DuckDuckGo search failed for query='%s': %s", cleaned, exc)
            return [], f"Web search is currently unavailable: {exc}"


class SafeWebPageReader:
    """SSRF-protected HTTP webpage fetcher and BeautifulSoup content extractor."""

    def __init__(
        self,
        timeout_sec: float = 12.0,
        max_slice_chars: int = _DEFAULT_PAGE_SLICE_CHARS,
    ) -> None:
        self.timeout_sec = float(timeout_sec)
        self.max_slice_chars = int(max_slice_chars)

    def extract_page(self, url: str) -> ExtractedWebPage:
        """Fetch an HTTPS URL safely (checking every redirect against SSRF) and extract structured text."""
        requested = (url or "").strip()
        try:
            current_url = validate_remote_url(
                requested,
                allow_insecure_http=False,
                allow_private_network=False,
            )
        except MCPSecurityError as exc:
            return ExtractedWebPage(
                requested_url=requested,
                final_url=requested,
                canonical_url=requested,
                title="",
                description="",
                content="",
                success=False,
                error=f"Blocked unsafe URL: {exc}",
            )

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1",
            "Accept-Language": "en-US,en;q=0.9",
        }
        response: Optional[httpx.Response] = None

        try:
            with httpx.Client(timeout=self.timeout_sec, follow_redirects=False, headers=headers) as client:
                for _ in range(_MAX_REDIRECT_HOPS + 1):
                    response = client.get(current_url)
                    if response.status_code in (301, 302, 303, 307, 308):
                        location = str(response.headers.get("location") or "").strip()
                        if not location:
                            break
                        next_url = urljoin(current_url, location)
                        try:
                            current_url = validate_remote_url(
                                next_url,
                                allow_insecure_http=False,
                                allow_private_network=False,
                            )
                        except MCPSecurityError as exc:
                            return ExtractedWebPage(
                                requested_url=requested,
                                final_url=next_url,
                                canonical_url=next_url,
                                title="",
                                description="",
                                content="",
                                status_code=int(response.status_code),
                                success=False,
                                error=f"Blocked unsafe URL redirect: {exc}",
                            )
                        continue
                    break
        except Exception as exc:
            return ExtractedWebPage(
                requested_url=requested,
                final_url=current_url,
                canonical_url=current_url,
                title="",
                description="",
                content="",
                success=False,
                error=f"HTTP fetch failed: {exc}",
            )

        assert response is not None
        status_code = int(getattr(response, "status_code", 0) or 0)
        if status_code >= 400:
            return ExtractedWebPage(
                requested_url=requested,
                final_url=current_url,
                canonical_url=current_url,
                title="",
                description="",
                content="",
                status_code=status_code,
                success=False,
                error=f"HTTP {status_code} while accessing webpage.",
            )

        content_type = str(response.headers.get("content-type") or "").lower()
        if content_type and not any(
            tok in content_type
            for tok in ("text/html", "application/xhtml", "text/plain", "application/json", "text/markdown")
        ):
            return ExtractedWebPage(
                requested_url=requested,
                final_url=current_url,
                canonical_url=current_url,
                title="",
                description="",
                content="",
                status_code=status_code,
                success=False,
                error=f"Unsupported content type '{content_type}'.",
            )

        raw_bytes = bytes(getattr(response, "content", b"") or b"")[:_MAX_HTML_BYTES]
        raw_text = raw_bytes.decode(getattr(response, "encoding", None) or "utf-8", errors="replace")

        if "html" in content_type or "<html" in raw_text[:600].lower():
            title, description, canonical_url, body, tables = extract_readable_html(
                raw_text,
                base_url=current_url,
            )
        else:
            title, description, canonical_url, tables = "", "", current_url, []
            body = raw_text.strip()[:_MAX_EXTRACTED_CHARS]

        if not body:
            return ExtractedWebPage(
                requested_url=requested,
                final_url=current_url,
                canonical_url=canonical_url,
                title=title,
                description=description,
                content="",
                tables=tables,
                status_code=status_code,
                success=False,
                error="Page loaded, but no readable text content could be extracted.",
            )

        clean_body = sanitize_tool_output(body)
        return ExtractedWebPage(
            requested_url=requested,
            final_url=current_url,
            canonical_url=canonical_url,
            title=title,
            description=description,
            content=clean_body,
            tables=tables,
            status_code=status_code,
            success=True,
            error="",
            char_count=len(clean_body),
        )

    def fetch_text(self, url: str, cursor: int = 0) -> str:
        """Fetch an HTTPS URL safely and return a clean paginated text view."""
        page = self.extract_page(url)
        if not page.success:
            if page.error.startswith("Blocked unsafe URL"):
                raise MCPSecurityError(page.error)
            return f"Unable to extract readable content from '{page.final_url}': {page.error}"

        start = max(0, int(cursor))
        end = start + self.max_slice_chars
        snippet = page.content[start:end]
        header = f"Title: {page.title}\n" if page.title else ""
        header += f"URL: {page.final_url}\n"
        if page.canonical_url and page.canonical_url != page.final_url:
            header += f"Canonical URL: {page.canonical_url}\n"
        header += f"Chars: {start}-{min(end, len(page.content))} of {len(page.content)}\n"
        if end < len(page.content):
            header += f"More content available at cursor={end}.\n"
        return sanitize_tool_output(f"{header}\n{snippet}")


def build_web_search_tool(
    searcher: Optional[DuckDuckGoWebSearcher] = None,
) -> BaseTool:
    """Create a LangChain-compatible web search tool bound to DuckDuckGoWebSearcher."""
    active_searcher = searcher or DuckDuckGoWebSearcher()

    @tool("web_search", args_schema=WebSearchInput)
    def web_search(query: str) -> str:
        """Search the public web and return numbered search results with result_id and URLs. Follow up with open_web_page to read full content."""
        results, error = active_searcher.search(query)
        if error:
            return error
        if not results:
            return "No relevant web search results were found."
        formatted = []
        for idx, hit in enumerate(results, start=1):
            url_line = f"\nURL: {hit.url}" if hit.url else ""
            formatted.append(f"[{idx}] {hit.title}{url_line}\n{hit.snippet}")
        return "\n\n".join(formatted)

    return web_search


def build_open_web_page_tool(
    searcher: Optional[DuckDuckGoWebSearcher] = None,
    reader: Optional[SafeWebPageReader] = None,
) -> BaseTool:
    """Create an SSRF-safe LangChain tool to open and read a webpage by URL or search result_id."""
    active_searcher = searcher or DuckDuckGoWebSearcher()
    active_reader = reader or SafeWebPageReader()

    @tool("open_web_page", args_schema=OpenWebPageInput)
    def open_web_page(
        url: str = "",
        result_id: Optional[int] = None,
        cursor: int = 0,
    ) -> str:
        """Open and read extracted text/tables from a public https:// URL or a 1-based result_id from web_search."""
        target_url = (url or "").strip()
        if not target_url and result_id is not None:
            hit = active_searcher.get_result_by_id(result_id)
            if hit is None or not hit.url:
                return f"No cached search result URL found for result_id={result_id}. Pass an explicit https:// url instead."
            target_url = hit.url

        if not target_url:
            return "Provide either an https:// url or a valid 1-based result_id from web_search."

        try:
            return active_reader.fetch_text(target_url, cursor=cursor)
        except MCPSecurityError as exc:
            return f"Blocked unsafe URL: {exc}"
        except Exception as exc:
            logger.warning("Failed to open web page '%s': %s", target_url, exc)
            return f"Unable to open web page '{target_url}': {exc}"

    return open_web_page
