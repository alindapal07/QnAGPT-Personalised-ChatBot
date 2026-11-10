"""Pytest suite for DuckDuckGo web search and user-scoped document tools."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from qna_bot.config.settings import AppSettings
from qna_bot.rag.ingestion import DocumentIngestor
from qna_bot.rag.retriever import DocumentRetriever
from qna_bot.rag.vector_store import FAISSVectorStoreManager
from qna_bot.tools.document_tools import build_document_tools
from qna_bot.tools.web_search import DuckDuckGoWebSearcher, build_web_search_tool
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
