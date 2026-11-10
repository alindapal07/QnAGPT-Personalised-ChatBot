"""LangChain tool and web research exports."""

from qna_bot.tools.document_tools import build_document_tools
from qna_bot.tools.web_research import (
    ResearchEntityRecord,
    ResearchExtractionSchema,
    cross_verify_research_records,
    fetch_and_extract_sources,
    filter_relevant_passages,
    format_opened_pages_for_prompt,
    format_verified_research_for_prompt,
    needs_structured_entity_research,
    parse_web_search_tool_text,
    select_relevant_sources,
)
from qna_bot.tools.web_search import (
    DuckDuckGoWebSearcher,
    ExtractedWebPage,
    SafeWebPageReader,
    WebSearchResult,
    build_open_web_page_tool,
    build_web_search_tool,
)

__all__ = [
    "DuckDuckGoWebSearcher",
    "ExtractedWebPage",
    "ResearchEntityRecord",
    "ResearchExtractionSchema",
    "SafeWebPageReader",
    "WebSearchResult",
    "build_document_tools",
    "build_open_web_page_tool",
    "build_web_search_tool",
    "cross_verify_research_records",
    "fetch_and_extract_sources",
    "filter_relevant_passages",
    "format_opened_pages_for_prompt",
    "format_verified_research_for_prompt",
    "needs_structured_entity_research",
    "parse_web_search_tool_text",
    "select_relevant_sources",
]
