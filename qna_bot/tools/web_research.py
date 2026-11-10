"""Multi-step web research pipeline: source selection, extraction, filtering, and cross-source verification."""

from concurrent.futures import ThreadPoolExecutor
import re
from typing import Any, Dict, List, Literal, Optional, Sequence
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from qna_bot.core.logging_config import get_logger
from qna_bot.tools.web_search import ExtractedWebPage, SafeWebPageReader

logger = get_logger(__name__)

_LOW_VALUE_DOMAINS = (
    "facebook.com",
    "instagram.com",
    "pinterest.com",
    "quora.com",
    "x.com",
    "twitter.com",
)
_ENTITY_RESEARCH_PATTERN = re.compile(
    r"\b("
    r"startups?|companies|firms|unicorns|organizations|"
    r"founders?|ceos?|executives?|team|"
    r"funding|raised|valuation|investors?|series\s+[a-d]|seed|"
    r"headquarters?|websites?|compare|comparison|table|"
    r"top\s+\d+|list\s+(?:of\s+)?\d*|find\s+(?:\d+|five|ten)"
    r")\b",
    flags=re.IGNORECASE,
)


class ResearchEntityRecord(BaseModel):
    """Structured organization/startup record extracted strictly from opened webpage text."""

    name: str = Field(..., min_length=1)
    founders: List[str] = Field(default_factory=list)
    website: Optional[str] = None
    funding_amount: Optional[str] = None
    funding_stage: Optional[str] = None
    headquarters: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)
    verification_status: Literal["confirmed", "reported", "unverified"] = "unverified"


class ResearchExtractionSchema(BaseModel):
    """Structured research bundle matching multi-entity queries."""

    companies: List[ResearchEntityRecord] = Field(default_factory=list)


def _query_keywords(text: str) -> set[str]:
    stop = {
        "the", "and", "for", "with", "that", "from", "this", "what", "when", "where",
        "which", "show", "find", "their", "about", "into", "list", "give", "using",
        "details", "top", "best", "first", "page", "open", "search",
    }
    return {
        tok
        for tok in re.findall(r"[a-z0-9]{2,}", (text or "").lower())
        if tok not in stop
    }


def needs_structured_entity_research(query: str) -> bool:
    """Return True when the query asks for multi-entity facts (founders, funding, websites, comparison)."""
    return bool(_ENTITY_RESEARCH_PATTERN.search(query or ""))


def select_relevant_sources(
    query: str,
    search_results: Sequence[Dict[str, Any]],
    max_sources: int = 6,
) -> List[Dict[str, Any]]:
    """Rank search result candidates by query alignment and domain accessibility likelihood."""
    q_tokens = _query_keywords(query)
    scored: List[tuple[tuple[int, int, int], Dict[str, Any]]] = []

    for idx, item in enumerate(search_results):
        url = str(item.get("url") or "").strip()
        if not url.startswith(("http://", "https://")):
            continue
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        domain = (urlparse(url).netloc or "").lower()

        text_tokens = _query_keywords(f"{title} {snippet} {domain}")
        overlap = len(q_tokens & text_tokens)
        domain_penalty = -3 if any(bad in domain for bad in _LOW_VALUE_DOMAINS) else 0
        https_bonus = 1 if url.startswith("https://") else 0
        # Preserve search ranking as tiebreaker (-idx)
        scored.append(((overlap + domain_penalty, https_bonus, -idx), dict(item)))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [pair[1] for pair in scored[:max_sources]]


def filter_relevant_passages(
    page_content: str,
    query: str,
    max_chars: int = 4500,
) -> str:
    """Filter extracted page text for the most informative, query-relevant sections and Markdown tables."""
    cleaned = (page_content or "").strip()
    if len(cleaned) <= max_chars:
        return cleaned

    q_tokens = _query_keywords(query)
    detail_markers = ("founder", "ceo", "funding", "raised", "million", "billion", "website", "http", "hq", "headquarter")
    blocks = [b.strip() for b in cleaned.split("\n\n") if b.strip()]
    if not blocks:
        return cleaned[:max_chars]

    scored_blocks: List[tuple[int, tuple[int, int], str]] = []
    for idx, block in enumerate(blocks):
        lower = block.lower()
        b_tokens = _query_keywords(block)
        overlap = len(q_tokens & b_tokens)
        table_bonus = 4 if (block.startswith("| ") and "\n| ---" in block) else 0
        detail_bonus = sum(1 for marker in detail_markers if marker in lower)
        intro_bonus = 3 if idx < 2 else 0
        score = (overlap * 2 + table_bonus + detail_bonus + intro_bonus, -idx)
        scored_blocks.append((idx, score, block))

    ranked = sorted(scored_blocks, key=lambda item: item[1], reverse=True)
    chosen_indices: set[int] = set()
    budget = 0
    for idx, _, block in ranked:
        projected = budget + len(block) + 2
        if projected > max_chars and chosen_indices:
            continue
        chosen_indices.add(idx)
        budget = projected
        if budget >= max_chars:
            break

    ordered = [block for idx, _, block in scored_blocks if idx in chosen_indices]
    result = "\n\n".join(ordered)
    return result[:max_chars].strip()


def fetch_and_extract_sources(
    search_results: Sequence[Dict[str, Any]],
    query: str,
    reader: Optional[SafeWebPageReader] = None,
    target_opened: int = 3,
    max_attempts: int = 6,
) -> tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Open ranked search results in parallel with graceful fallback when pages fail or block bots."""
    active_reader = reader or SafeWebPageReader()
    ranked_candidates = select_relevant_sources(
        query=query,
        search_results=search_results,
        max_sources=max_attempts,
    )
    if not ranked_candidates:
        return [], [
            {
                "title": str(r.get("title") or "Web Result"),
                "url": str(r.get("url") or ""),
                "snippet": str(r.get("snippet") or ""),
                "status": "snippet_only",
            }
            for r in search_results
        ]

    opened_pages: List[Dict[str, Any]] = []
    status_by_url: Dict[str, Dict[str, Any]] = {}

    batch_size = max(target_opened, min(4, len(ranked_candidates)))
    first_batch = ranked_candidates[:batch_size]
    fallback_batch = ranked_candidates[batch_size:]

    def _fetch_one(candidate: Dict[str, Any]) -> tuple[Dict[str, Any], ExtractedWebPage]:
        url = str(candidate.get("url") or "").strip()
        return candidate, active_reader.extract_page(url)

    def _process_pair(candidate: Dict[str, Any], page: ExtractedWebPage) -> None:
        url = str(candidate.get("url") or "").strip()
        if page.success and page.content:
            filtered = filter_relevant_passages(page.content, query=query, max_chars=4200)
            entry = {
                "url": page.final_url or url,
                "requested_url": url,
                "canonical_url": page.canonical_url or page.final_url or url,
                "title": page.title or str(candidate.get("title") or "Web Page"),
                "description": page.description,
                "content": filtered,
                "tables": page.tables[:3],
                "status_code": page.status_code,
                "success": True,
                "error": "",
                "char_count": page.char_count,
            }
            opened_pages.append(entry)
            status_by_url[url] = {
                "status": "read",
                "error": "",
                "excerpt": filtered[:280].replace("\n", " "),
                "final_url": page.final_url or url,
                "title": entry["title"],
            }
        else:
            opened_pages.append(
                {
                    "url": page.final_url or url,
                    "requested_url": url,
                    "canonical_url": page.canonical_url or url,
                    "title": str(candidate.get("title") or "Web Page"),
                    "description": "",
                    "content": "",
                    "tables": [],
                    "status_code": page.status_code,
                    "success": False,
                    "error": page.error or "Could not extract readable content.",
                    "char_count": 0,
                }
            )
            status_by_url[url] = {
                "status": "inaccessible",
                "error": page.error or "Page inaccessible",
                "excerpt": str(candidate.get("snippet") or ""),
                "final_url": url,
                "title": str(candidate.get("title") or "Web Result"),
            }

    with ThreadPoolExecutor(max_workers=min(4, len(first_batch))) as pool:
        for candidate, page in pool.map(_fetch_one, first_batch):
            _process_pair(candidate, page)

    successful_count = sum(1 for p in opened_pages if p.get("success"))
    for candidate in fallback_batch:
        if successful_count >= target_opened:
            break
        cand, page = _fetch_one(candidate)
        _process_pair(cand, page)
        if page.success and page.content:
            successful_count += 1

    updated_sources: List[Dict[str, str]] = []
    for raw in search_results:
        url = str(raw.get("url") or "").strip()
        info = status_by_url.get(url)
        if info:
            updated_sources.append(
                {
                    "title": str(raw.get("title") or info.get("title") or "Web Result"),
                    "url": info.get("final_url") or url,
                    "snippet": str(info.get("excerpt") or raw.get("snippet") or ""),
                    "status": str(info.get("status") or "snippet_only"),
                    "error": str(info.get("error") or ""),
                }
            )
        else:
            updated_sources.append(
                {
                    "title": str(raw.get("title") or "Web Result"),
                    "url": url,
                    "snippet": str(raw.get("snippet") or ""),
                    "status": "snippet_only",
                    "error": "",
                }
            )

    return opened_pages, updated_sources


def _clean_nullable_value(value: Optional[str]) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    if text.lower() in {"none", "null", "n/a", "na", "unknown", "not available", "not publicly available"}:
        return None
    return text


def _domain_of(url: Optional[str]) -> str:
    if not url:
        return ""
    host = (urlparse(url).netloc or "").lower()
    return host[4:] if host.startswith("www.") else host


def cross_verify_research_records(
    records: Sequence[ResearchEntityRecord],
    opened_pages: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Deduplicate entities across sources, detect conflicts, and assign verification statuses."""
    valid_source_urls = {
        str(p.get("url") or "").strip()
        for p in opened_pages
        if p.get("success")
    } | {
        str(p.get("requested_url") or "").strip()
        for p in opened_pages
        if p.get("success")
    }
    corpus_text = "\n".join(
        str(p.get("content") or "").lower()
        for p in opened_pages
        if p.get("success")
    )

    merged: Dict[str, Dict[str, Any]] = {}
    conflict_notes: List[str] = []

    for rec in records:
        name_clean = re.sub(r"\s+", " ", rec.name or "").strip()
        if not name_clean:
            continue
        key = re.sub(r"[^a-z0-9]+", "", name_clean.lower())
        if not key:
            continue

        # Only keep source URLs that were actually retrieved cleanly if known
        filtered_sources = [
            s.strip()
            for s in (rec.source_urls or [])
            if s.strip() and (not valid_source_urls or s.strip() in valid_source_urls)
        ]
        founders_clean = [
            f.strip()
            for f in (rec.founders or [])
            if f.strip() and f.strip().lower() not in {"unknown", "n/a", "none", "not publicly available"}
        ]
        website = _clean_nullable_value(rec.website)
        funding_amount = _clean_nullable_value(rec.funding_amount)
        funding_stage = _clean_nullable_value(rec.funding_stage)
        hq = _clean_nullable_value(rec.headquarters)

        # Ground funding mentions against retrieved page corpus to reduce hallucination risk
        if funding_amount and corpus_text:
            num_tokens = re.findall(r"\d+(?:\.\d+)?", funding_amount)
            if num_tokens and not any(tok in corpus_text for tok in num_tokens):
                funding_amount = None

        if key not in merged:
            merged[key] = {
                "name": name_clean,
                "founders": list(dict.fromkeys(founders_clean)),
                "website": website,
                "funding_amount": funding_amount,
                "funding_stage": funding_stage,
                "headquarters": hq,
                "source_urls": list(dict.fromkeys(filtered_sources)),
                "verification_status": "unverified",
            }
            continue

        existing = merged[key]
        for f in founders_clean:
            if f not in existing["founders"]:
                existing["founders"].append(f)
        if not existing["website"] and website:
            existing["website"] = website
        elif existing["website"] and website and _domain_of(existing["website"]) != _domain_of(website):
            conflict_notes.append(
                f"{existing['name']}: conflicting websites ({existing['website']} vs {website})."
            )
        if not existing["funding_amount"] and funding_amount:
            existing["funding_amount"] = funding_amount
        elif existing["funding_amount"] and funding_amount and existing["funding_amount"].lower() != funding_amount.lower():
            conflict_notes.append(
                f"{existing['name']}: differing reported funding ({existing['funding_amount']} vs {funding_amount})."
            )
        if not existing["funding_stage"] and funding_stage:
            existing["funding_stage"] = funding_stage
        if not existing["headquarters"] and hq:
            existing["headquarters"] = hq
        for src in filtered_sources:
            if src not in existing["source_urls"]:
                existing["source_urls"].append(src)

    companies: List[Dict[str, Any]] = []
    for entry in merged.values():
        distinct_domains = {_domain_of(u) for u in entry["source_urls"] if _domain_of(u)}
        website_domain = _domain_of(entry["website"])
        has_primary_site = bool(website_domain and website_domain in distinct_domains)
        if len(distinct_domains) >= 2 or has_primary_site:
            entry["verification_status"] = "confirmed"
        elif entry["source_urls"]:
            entry["verification_status"] = "reported"
        else:
            entry["verification_status"] = "unverified"
        companies.append(entry)

    return {
        "companies": companies,
        "conflicts": conflict_notes,
        "opened_source_count": sum(1 for p in opened_pages if p.get("success")),
    }


def format_opened_pages_for_prompt(
    opened_pages: Sequence[Dict[str, Any]],
    fallback_snippets: Sequence[Dict[str, Any]] = (),
) -> str:
    """Format full extracted webpage content cleanly with explicit attribution and failure notes."""
    blocks: List[str] = []
    successful_urls: set[str] = set()
    failed_notes: List[str] = []

    idx_counter = 1
    for page in opened_pages:
        url = str(page.get("url") or "").strip()
        title = str(page.get("title") or "Web Page").strip()
        if page.get("success") and page.get("content"):
            successful_urls.add(url)
            if page.get("requested_url"):
                successful_urls.add(str(page["requested_url"]))
            blocks.append(
                f"[Opened Webpage {idx_counter}: {title}]\n"
                f"URL: {url}\n"
                f"Extracted Content:\n{page.get('content', '')}"
            )
            idx_counter += 1
        else:
            err = str(page.get("error") or "Inaccessible").strip()
            failed_notes.append(f"- {title} ({url}): {err}")

    snippet_blocks: List[str] = []
    for raw in fallback_snippets:
        url = str(raw.get("url") or "").strip()
        if url in successful_urls:
            continue
        snippet = str(raw.get("snippet") or "").strip()
        if snippet:
            snippet_blocks.append(
                f"[Search Snippet Only (Page Not Opened): {raw.get('title', 'Web Result')}]"
                + (f" (URL: {url})" if url else "")
                + f"\n{snippet}"
            )

    sections: List[str] = []
    if blocks:
        sections.append("\n\n---\n\n".join(blocks))
    if failed_notes:
        sections.append("Inaccessible Sources (Fallback Used):\n" + "\n".join(failed_notes))
    if snippet_blocks:
        sections.append("Additional Unopened Search Snippets:\n" + "\n\n".join(snippet_blocks[:4]))
    return "\n\n".join(sections)


def format_verified_research_for_prompt(verified_research: Dict[str, Any]) -> str:
    """Render structured cross-verified entity research cleanly for final answer synthesis."""
    companies = verified_research.get("companies") or []
    if not companies:
        return ""
    lines: List[str] = ["Structured Cross-Verified Facts (use 'Not publicly available' for nulls):"]
    for idx, comp in enumerate(companies, start=1):
        founders = ", ".join(comp.get("founders") or []) or "Not publicly available"
        website = comp.get("website") or "Not publicly available"
        funding = comp.get("funding_amount") or "Not publicly available"
        stage = f" ({comp['funding_stage']})" if comp.get("funding_stage") else ""
        hq = comp.get("headquarters") or "Not publicly available"
        sources = ", ".join(comp.get("source_urls") or []) or "None"
        lines.append(
            f"{idx}. {comp['name']} [{comp.get('verification_status', 'unverified')}] | "
            f"Founders: {founders} | Website: {website} | Funding: {funding}{stage} | "
            f"HQ: {hq} | Sources: {sources}"
        )
    conflicts = verified_research.get("conflicts") or []
    if conflicts:
        lines.append("Conflicts noted across sources:\n- " + "\n- ".join(conflicts))
    return "\n".join(lines)


def parse_web_search_tool_text(tool_output: str) -> List[Dict[str, str]]:
    """Extract {title, url, snippet} items back out of a web_search tool response string for UI citations."""
    results: List[Dict[str, str]] = []
    for block in re.split(r"\n\n+", (tool_output or "").strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines or not lines[0].startswith("["):
            continue
        title = re.sub(r"^\[\d+\]\s*", "", lines[0]).strip()
        url = ""
        snippet_parts: List[str] = []
        for line in lines[1:]:
            if line.startswith("URL:"):
                url = line[4:].strip()
            else:
                snippet_parts.append(line)
        if title or url:
            results.append(
                {
                    "title": title or "Web Result",
                    "url": url,
                    "snippet": " ".join(snippet_parts),
                    "status": "snippet_only",
                    "error": "",
                }
            )
    return results
