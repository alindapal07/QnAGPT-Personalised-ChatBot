"""Web search, webpage opening, and research cross-verification nodes."""

import re
import time
from typing import Any, Dict, List, Optional
import uuid

from qna_bot.agent.nodes.base import BaseNodeSuite, logger
from qna_bot.agent.prompts import RESEARCH_EXTRACTION_PROMPT
from qna_bot.agent.state import AgentState
from qna_bot.tools.web_research import (
    ResearchExtractionSchema,
    cross_verify_research_records,
    fetch_and_extract_sources,
    format_opened_pages_for_prompt,
)


class WebNodesMixin:
    """Mixin containing web search, webpage fetching, and cross-verification nodes."""

    def web_search(self: BaseNodeSuite, state: AgentState) -> Dict[str, Any]:
        """Execute DuckDuckGo search and preserve structured web URLs."""
        query = (state.get("rewritten_query") or state.get("user_query") or "").strip()
        results, err = self.web_searcher.search(query)
        serialized = [r.to_dict() for r in results]
        return {
            "web_results": serialized,
            "web_sources": serialized,
            "error": err or state.get("error", ""),
            "activity": "Searching the web...",
        }

    def refine_web_search(self: BaseNodeSuite, state: AgentState) -> Dict[str, Any]:
        """Broaden or focus the search query cleanly when initial results/pages were insufficient."""
        original = (state.get("rewritten_query") or state.get("user_query") or "").strip()
        retries = int(state.get("web_retry_count", 0))
        cleaned = re.sub(
            r"\b(please|can you|show me|give me|tell me|find|and show their|details|open the first result)\b",
            " ",
            original,
            flags=re.IGNORECASE,
        )
        words = [w for w in cleaned.split() if len(w) > 1]
        refined = " ".join(words[:10]) if words else original
        if refined.lower() == original.lower():
            refined = f"{original} overview official list"
        return {
            "rewritten_query": refined.strip(),
            "web_retry_count": retries + 1,
            "activity": "Refining web search query...",
        }

    def open_webpages(self: BaseNodeSuite, state: AgentState) -> Dict[str, Any]:
        """Open selected search results, extract clean content/tables, and fall back if any page fails."""
        query = (state.get("user_query") or state.get("rewritten_query") or "").strip()
        web_results = state.get("web_results") or []
        opened_pages, updated_sources = fetch_and_extract_sources(
            search_results=web_results,
            query=query,
            reader=self.page_reader,
            target_opened=3,
            max_attempts=6,
        )

        new_execs: List[Dict[str, Any]] = []
        now = time.time()
        for page in opened_pages:
            new_execs.append(
                {
                    "execution_id": f"exec_{uuid.uuid4().hex[:10]}",
                    "server_name": "QnAGPT Web Research",
                    "tool_name": "open_web_page",
                    "status": "completed" if page.get("success") else "failed",
                    "summary": (
                        f"{page.get('title', 'Web Page')} ({page.get('char_count', 0)} chars) — {page.get('url', '')}"
                        if page.get("success")
                        else f"Failed {page.get('url', '')}: {page.get('error', 'inaccessible')}"
                    )[:280],
                    "duration_ms": 0.0,
                    "timestamp": now,
                }
            )

        existing_execs = list(state.get("tool_executions") or [])
        result_payload: Dict[str, Any] = {
            "opened_webpages": opened_pages,
            "web_sources": updated_sources or list(state.get("web_sources") or []),
            "tool_executions": existing_execs + new_execs,
            "activity": "Reading webpage content...",
        }
        if getattr(self.settings, "crag_enabled", True):
            web_bundle = self.crag_pipeline.assemble_web_evidence(
                web_results=web_results,
                opened_pages=opened_pages,
                query=query,
            )
            result_payload["evidence_bundle"] = web_bundle.to_dict()
            result_payload["crag_status"] = web_bundle.status.value
            result_payload["sources"] = web_bundle.sources
        return result_payload

    def extract_and_verify(self: BaseNodeSuite, state: AgentState) -> Dict[str, Any]:
        """Extract structured entity records strictly from opened webpages and cross-verify across sources."""
        query = (state.get("user_query") or "").strip()
        opened_pages = [p for p in (state.get("opened_webpages") or []) if p.get("success") and p.get("content")]
        if not opened_pages or not self.api_key:
            return {"verified_research": {}, "activity": "Verifying extracted facts..."}

        page_context = format_opened_pages_for_prompt(opened_pages)
        prompt = (
            f"{RESEARCH_EXTRACTION_PROMPT}\n\n"
            f"User Question: {query}\n\n"
            f"EXTRACTED WEBPAGE CONTENT (UNTRUSTED DATA):\n{page_context[:14000]}"
        )
        extracted: Optional[ResearchExtractionSchema] = None
        try:
            candidate = self._make_llm(temperature=0.0, max_tokens=1536).with_structured_output(
                ResearchExtractionSchema
            ).invoke(prompt)
            if isinstance(candidate, ResearchExtractionSchema):
                extracted = candidate
        except Exception as exc:
            logger.warning("Structured research extraction failed cleanly: %s", exc)

        if extracted is None or not extracted.companies:
            return {
                "verified_research": {"companies": [], "conflicts": [], "opened_source_count": len(opened_pages)},
                "activity": "Verifying extracted facts...",
            }

        verified = cross_verify_research_records(extracted.companies, opened_pages)
        return {
            "verified_research": verified,
            "activity": "Cross-verifying sources...",
        }
