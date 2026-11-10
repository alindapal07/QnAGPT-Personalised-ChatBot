"""DuckDuckGo web search service and LangChain tool."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from qna_bot.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class WebSearchResult:
    """Structured web search hit preserving source URL."""

    title: str
    url: str
    snippet: str

    def to_dict(self) -> Dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


class WebSearchInput(BaseModel):
    """Typed input schema for the DuckDuckGo web search tool."""

    query: str = Field(..., min_length=1, description="Concise web search query.")


class DuckDuckGoWebSearcher:
    """Wrapper around DuckDuckGo Search returning structured results with URLs."""

    def __init__(self, max_results: int = 4) -> None:
        self.max_results = max_results

    def search(self, query: str) -> tuple[List[WebSearchResult], str]:
        """Run a DuckDuckGo search and return (results, error_message)."""
        cleaned = (query or "").strip()
        if not cleaned:
            return [], "Empty web search query."

        try:
            wrapper = DuckDuckGoSearchAPIWrapper(max_results=self.max_results)
            raw_items: List[Dict[str, Any]] = wrapper.results(
                cleaned,
                max_results=self.max_results,
            )
            results: List[WebSearchResult] = []
            for item in raw_items or []:
                title = str(item.get("title") or "Web Result").strip()
                url = str(item.get("link") or item.get("url") or "").strip()
                snippet = str(item.get("snippet") or item.get("body") or "").strip()
                if snippet or url:
                    results.append(WebSearchResult(title=title, url=url, snippet=snippet))

            if results:
                return results, ""

            # Fallback to DuckDuckGoSearchRun if structured wrapper returned empty
            runner = DuckDuckGoSearchRun(api_wrapper=wrapper)
            text_out = str(runner.invoke(cleaned) or "").strip()
            if text_out and "No good DuckDuckGo Search Result" not in text_out:
                return [WebSearchResult(title="DuckDuckGo Search", url="", snippet=text_out)], ""
            return [], ""
        except Exception as exc:
            logger.warning("DuckDuckGo search failed for query='%s': %s", cleaned, exc)
            return [], f"Web search is currently unavailable: {exc}"


def build_web_search_tool(
    searcher: Optional[DuckDuckGoWebSearcher] = None,
) -> BaseTool:
    """Create a LangChain-compatible web search tool bound to DuckDuckGoWebSearcher."""
    active_searcher = searcher or DuckDuckGoWebSearcher()

    @tool("web_search", args_schema=WebSearchInput)
    def web_search(query: str) -> str:
        """Search the public web via DuckDuckGo for recent events, developments, or external facts."""
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
