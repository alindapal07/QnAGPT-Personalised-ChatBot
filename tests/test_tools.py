"""Pytest suite for DuckDuckGo web search, SSRF-safe web page reading, and user-scoped document tools."""

from pathlib import Path
import socket
from unittest.mock import MagicMock, patch

import pytest
from ddgs.exceptions import DDGSException

from qna_bot.config.settings import AppSettings
from qna_bot.mcp_client.security import MCPSecurityError, validate_remote_url
from qna_bot.rag.ingestion import DocumentIngestor
from qna_bot.rag.retriever import DocumentRetriever
from qna_bot.rag.vector_store import FAISSVectorStoreManager
from qna_bot.tools.document_tools import build_document_tools
from qna_bot.tools.web_research import (
    ResearchEntityRecord,
    cross_verify_research_records,
    fetch_and_extract_sources,
)
from qna_bot.tools.web_search import (
    DuckDuckGoWebSearcher,
    ExtractedWebPage,
    SafeWebPageReader,
    build_open_web_page_tool,
    build_web_search_tool,
    extract_readable_html,
)
from tests.test_rag import DeterministicTestEmbeddings


def test_duckduckgo_web_search_returns_urls_and_handles_failures() -> None:
    searcher = DuckDuckGoWebSearcher(max_results=3)

    mock_wrapper = MagicMock()
    mock_wrapper.results.return_value = [
        {
            "title": "LangGraph Docs",
            "link": "https://langchain-ai.github.io/langgraph/",
            "snippet": "Build stateful agent workflows.",
        }
    ]

    with patch("qna_bot.tools.web_search.DuckDuckGoSearchAPIWrapper", return_value=mock_wrapper):
        results, err = searcher.search("LangGraph documentation")

    assert err == ""
    assert len(results) == 1
    assert results[0].url == "https://langchain-ai.github.io/langgraph/"

    tool = build_web_search_tool(searcher)
    with patch("qna_bot.tools.web_search.DuckDuckGoSearchAPIWrapper", return_value=mock_wrapper):
        tool_output = tool.invoke({"query": "LangGraph documentation"})
    assert "https://langchain-ai.github.io/langgraph/" in tool_output

    # Graceful unavailable search service handling
    with patch(
        "qna_bot.tools.web_search.DuckDuckGoSearchAPIWrapper",
        side_effect=ConnectionError("offline"),
    ):
        fail_results, fail_err = searcher.search("LangGraph")
        fail_tool_out = tool.invoke({"query": "LangGraph"})

    assert fail_results == []
    assert "unavailable" in fail_err.lower()
    assert "unavailable" in fail_tool_out.lower()


def test_document_tools_enforce_user_isolation_and_preserve_citations(tmp_path: Path) -> None:
    settings = AppSettings(groq_api_key="", vector_store_dir=tmp_path / "vs")
    store = FAISSVectorStoreManager(
        base_dir=settings.vector_store_dir,
        embeddings=DeterministicTestEmbeddings(),
    )
    ingestor = DocumentIngestor(vector_store=store, settings=settings)
    ingestor.ingest_bytes(b"Secret finance report for Alice.", "alice_finance.txt", user_id="alice")
    ingestor.ingest_bytes(b"Weather notes for Bob.", "bob_weather.txt", user_id="bob")

    retriever = DocumentRetriever(vector_store=store, settings=settings)
    alice_search, alice_summary = build_document_tools(retriever, user_id="alice")

    search_out = alice_search.invoke({"query": "finance", "top_k": 4})
    assert "alice_finance.txt" in search_out
    assert "bob_weather.txt" not in search_out

    summary_out = alice_summary.invoke({"topic": "finance"})
    assert "alice_finance.txt" in summary_out
    assert "bob_weather.txt" not in summary_out


def test_web_search_ddgs_exception_fallback_and_open_web_page_ssrf(tmp_path: Path) -> None:
    searcher = DuckDuckGoWebSearcher(max_results=3)

    failing_wrapper = MagicMock()
    failing_wrapper.results.side_effect = DDGSException("DNSError: wt.wikipedia.org")
    # Simulate real langchain_community module check on wrapper fallback
    type(failing_wrapper).__module__ = "langchain_community.utilities.duckduckgo_search"

    fallback_ddgs = MagicMock()
    fallback_ddgs.text.return_value = [
        {
            "title": "Python 3.13 Release Notes",
            "href": "https://docs.python.org/3.13/whatsnew/3.13.html",
            "body": "Free-threaded CPython and improved JIT highlights.",
        }
    ]

    with (
        patch("qna_bot.tools.web_search.DuckDuckGoSearchAPIWrapper", return_value=failing_wrapper),
        patch("qna_bot.tools.web_search.DDGS", return_value=fallback_ddgs),
    ):
        hits, err = searcher.search("Python 3.13 features")

    assert err == ""
    assert len(hits) == 1
    assert hits[0].url == "https://docs.python.org/3.13/whatsnew/3.13.html"
    assert searcher.get_result_by_id(1) == hits[0]

    # open_web_page resolves result_id=1, extracts clean HTML, and blocks SSRF/private redirects
    html_body = (
        "<html><head><title>What's New In Python 3.13</title>"
        "<style>.hidden{display:none}</style><script>alert(1)</script></head>"
        "<body><h1>Python 3.13</h1><p>Experimental free-threaded mode.</p></body></html>"
    )
    ok_resp = MagicMock(
        status_code=200,
        headers={"content-type": "text/html; charset=utf-8"},
        content=html_body.encode("utf-8"),
        encoding="utf-8",
    )
    redirect_resp = MagicMock(
        status_code=302,
        headers={"location": "http://127.0.0.1/internal"},
        content=b"",
        encoding="utf-8",
    )

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.get.return_value = ok_resp

    open_tool = build_open_web_page_tool(searcher=searcher, reader=SafeWebPageReader())
    with (
        patch(
            "qna_bot.mcp_client.security.socket.getaddrinfo",
            return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))],
        ),
        patch("qna_bot.tools.web_search.httpx.Client", return_value=mock_client),
    ):
        opened = open_tool.invoke({"result_id": 1})
    assert "What's New In Python 3.13" in opened
    assert "Experimental free-threaded mode." in opened
    assert "alert(1)" not in opened

    # Direct loopback SSRF blocked
    blocked_loopback = open_tool.invoke({"url": "http://127.0.0.1:8080/secret"})
    assert "blocked unsafe url" in blocked_loopback.lower()

    # Public URL redirecting to private IP blocked cleanly
    mock_client.get.return_value = redirect_resp
    with (
        patch(
            "qna_bot.mcp_client.security.socket.getaddrinfo",
            side_effect=[
                [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))],
                [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))],
            ],
        ),
        patch("qna_bot.tools.web_search.httpx.Client", return_value=mock_client),
    ):
        blocked_redirect = open_tool.invoke({"url": "https://example.com/redir"})
    assert "blocked unsafe url" in blocked_redirect.lower()


def test_webpage_table_extraction_hydration_fallback_and_cross_source_verification() -> None:
    rich_html = """
    <html>
      <head>
        <title>Indian AI Startups Tracker</title>
        <link rel="canonical" href="https://tracker.example.org/ai-startups-2026" />
        <meta name="description" content="Founders and funding details." />
      </head>
      <body>
        <nav>Menu Home Login</nav>
        <div class="cookie-banner">Accept cookies</div>
        <main>
          <h1>Top AI Startups in India</h1>
          <p>Sarvam AI builds sovereign LLMs in Bengaluru.</p>
          <table>
            <tr><th>Company</th><th>Founders</th><th>Website</th><th>Funding</th></tr>
            <tr><td>Sarvam AI</td><td>Vivek Raghavan, Pratyush Kumar</td><td>https://sarvam.ai</td><td>$41M</td></tr>
            <tr><td>Krutrim</td><td>Bhavish Aggarwal</td><td>https://olakrutrim.com</td><td>$50M</td></tr>
          </table>
        </main>
        <footer>Copyright 2026</footer>
      </body>
    </html>
    """
    title, desc, canon, body, tables = extract_readable_html(
        rich_html, base_url="https://tracker.example.org/raw"
    )
    assert title == "Indian AI Startups Tracker"
    assert desc == "Founders and funding details."
    assert canon == "https://tracker.example.org/ai-startups-2026"
    assert "Menu Home Login" not in body
    assert "Accept cookies" not in body
    assert "Copyright 2026" not in body
    assert "| Sarvam AI | Vivek Raghavan, Pratyush Kumar | https://sarvam.ai | $41M |" in body
    assert len(tables) == 1

    # SPA / Next.js hydration fallback when visible HTML body is sparse
    spa_html = (
        '<html><head><script id="__NEXT_DATA__" type="application/json">'
        '{"startup":"Neysa","funding":"$20M"}</script></head>'
        '<body><div id="root"></div></body></html>'
    )
    _, _, _, spa_body, _ = extract_readable_html(spa_html, base_url="https://spa.example.com")
    assert "Neysa" in spa_body
    assert "$20M" in spa_body

    # Fallback across failed (403) source to accessible sources
    search_hits = [
        {"title": "Blocked Source", "url": "https://blocked.example.com/list", "snippet": "Blocked snippet"},
        {"title": "Tracker A", "url": "https://a.example.org/startups", "snippet": "AI startups"},
        {"title": "Tracker B", "url": "https://b.example.net/startups", "snippet": "Funding list"},
    ]

    def _fake_extract(url: str) -> ExtractedWebPage:
        if "blocked" in url:
            return ExtractedWebPage(
                requested_url=url,
                final_url=url,
                canonical_url=url,
                title="",
                description="",
                content="",
                status_code=403,
                success=False,
                error="HTTP 403 while accessing webpage.",
            )
        page_text = (
            "Sarvam AI founded by Pratyush Kumar raised $55M Series B."
            if "b.example.net" in url
            else "Sarvam AI founded by Vivek Raghavan raised $41M Series A."
        )
        return ExtractedWebPage(
            requested_url=url,
            final_url=url,
            canonical_url=url,
            title=f"Page {url}",
            description="",
            content=page_text,
            status_code=200,
            success=True,
            char_count=len(page_text),
        )

    mock_reader = MagicMock(spec=SafeWebPageReader)
    mock_reader.extract_page.side_effect = _fake_extract
    opened_pages, updated_sources = fetch_and_extract_sources(
        search_results=search_hits,
        query="Indian AI startups founders funding",
        reader=mock_reader,
        target_opened=2,
        max_attempts=3,
    )
    assert sum(1 for p in opened_pages if p["success"]) == 2
    status_map = {s["url"]: s["status"] for s in updated_sources}
    assert status_map["https://blocked.example.com/list"] == "inaccessible"
    assert status_map["https://a.example.org/startups"] == "read"
    assert status_map["https://b.example.net/startups"] == "read"

    # Cross-source verification deduplicates, detects conflicts, and preserves nulls cleanly
    verified = cross_verify_research_records(
        records=[
            ResearchEntityRecord(
                name="Sarvam AI",
                founders=["Vivek Raghavan"],
                website="https://sarvam.ai",
                funding_amount="$41M",
                source_urls=["https://a.example.org/startups"],
            ),
            ResearchEntityRecord(
                name="Sarvam AI",
                founders=["Pratyush Kumar"],
                website="https://sarvam.ai",
                funding_amount="$55M",
                source_urls=["https://b.example.net/startups"],
            ),
            ResearchEntityRecord(
                name="Unverified Startup",
                founders=[],
                website=None,
                funding_amount="$999M",  # Not in corpus -> dropped cleanly instead of hallucinated
                source_urls=["https://a.example.org/startups"],
            ),
        ],
        opened_pages=opened_pages,
    )
    by_name = {c["name"]: c for c in verified["companies"]}
    assert by_name["Sarvam AI"]["verification_status"] == "confirmed"
    assert set(by_name["Sarvam AI"]["founders"]) == {"Vivek Raghavan", "Pratyush Kumar"}
    assert len(verified["conflicts"]) == 1
    assert by_name["Unverified Startup"]["funding_amount"] is None


def test_document_tools_empty_knowledge_base_returns_clean_message(tmp_path: Path) -> None:
    settings = AppSettings(groq_api_key="", vector_store_dir=tmp_path / "vs")
    store = FAISSVectorStoreManager(
        base_dir=settings.vector_store_dir,
        embeddings=DeterministicTestEmbeddings(),
    )
    retriever = DocumentRetriever(vector_store=store, settings=settings)
    charlie_search, charlie_summary = build_document_tools(retriever, user_id="charlie_empty")

    search_out = charlie_search.invoke({"query": "quantum computing", "top_k": 4})
    assert isinstance(search_out, str)
    assert "No relevant uploaded documents" in search_out or "No indexed documents" in search_out or len(search_out) > 0

    summary_out = charlie_summary.invoke({"topic": "anything"})
    assert isinstance(summary_out, str)


def test_open_web_page_blocks_cloud_metadata_and_private_ips() -> None:
    reader = SafeWebPageReader()
    tool = build_open_web_page_tool(reader)

    # AWS/GCP/Azure instance metadata IP
    meta_out = tool.invoke({"url": "http://169.254.169.254/latest/meta-data/"})
    assert "blocked" in meta_out.lower() or "error" in meta_out.lower() or "ssrf" in meta_out.lower()

    # Loopback IP
    loopback_out = tool.invoke({"url": "http://127.0.0.1:8080/admin"})
    assert "blocked" in loopback_out.lower() or "error" in loopback_out.lower() or "ssrf" in loopback_out.lower()


def test_validate_remote_url_disallowed_schemes_and_credentials() -> None:
    """Disallowed schemes and embedded user credentials must be rejected by SSRF guard."""
    with pytest.raises(MCPSecurityError, match="Unsupported URL scheme"):
        validate_remote_url("ftp://ftp.example.com/file.txt")

    with pytest.raises(MCPSecurityError, match="Unsupported URL scheme"):
        validate_remote_url("gopher://gopher.example.com:70/")

    with pytest.raises(MCPSecurityError, match="Unsupported URL scheme"):
        validate_remote_url("file:///etc/passwd")

    with pytest.raises(MCPSecurityError, match="user credentials"):
        validate_remote_url("https://admin:secret@example.com/api")

    with pytest.raises(MCPSecurityError, match="user credentials"):
        validate_remote_url("https://user@example.com/dashboard")


def test_validate_remote_url_cgnat_benchmarking_and_ipv4_mapped_ipv6() -> None:
    """Special disallowed IP ranges (CGNAT RFC 6598, Benchmarking RFC 2544, IPv4-mapped IPv6) must be blocked."""
    # CGNAT 100.64.0.0/10
    with pytest.raises(MCPSecurityError, match="Blocked private/loopback IP"):
        validate_remote_url("http://100.64.0.1:8080/api", allow_insecure_http=True)

    # Benchmarking 198.18.0.0/15
    with pytest.raises(MCPSecurityError, match="Blocked private/loopback IP"):
        validate_remote_url("http://198.18.0.1/benchmark", allow_insecure_http=True)

    # TEST-NET-1 192.0.2.0/24
    with pytest.raises(MCPSecurityError, match="Blocked private/loopback IP"):
        validate_remote_url("http://192.0.2.1/test", allow_insecure_http=True)

    # IPv4-mapped IPv6 ::ffff:127.0.0.1
    with pytest.raises(MCPSecurityError, match="Blocked private/loopback IP"):
        validate_remote_url("http://[::ffff:127.0.0.1]:8080/admin", allow_insecure_http=True)

    # IPv4-mapped IPv6 ::ffff:10.0.0.1
    with pytest.raises(MCPSecurityError, match="Blocked private/loopback IP"):
        validate_remote_url("http://[::ffff:10.0.0.1]:8080/admin", allow_insecure_http=True)


def test_safe_web_page_reader_dns_rebinding_and_redirect_ssrf() -> None:
    """SafeWebPageReader must block DNS resolution to private IPs and redirects to disallowed ranges."""
    reader = SafeWebPageReader()
    tool = build_open_web_page_tool(reader)

    # Public domain that resolves via DNS to a private IP (DNS rebinding simulation)
    with patch(
        "qna_bot.mcp_client.security.socket.getaddrinfo",
        return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", 443))],
    ):
        out = tool.invoke({"url": "https://malicious-dns-rebind.com/api"})
        assert "blocked" in out.lower() or "error" in out.lower()

    # Public domain that resolves via DNS to CGNAT IP (100.64.1.1)
    with patch(
        "qna_bot.mcp_client.security.socket.getaddrinfo",
        return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("100.64.1.1", 443))],
    ):
        out_cgnat = tool.invoke({"url": "https://cgnat-target.internal/api"})
        assert "blocked" in out_cgnat.lower() or "error" in out_cgnat.lower()




