"""Pytest suite for context-aware dynamic SuggestionService and caching."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from qna_bot.agent.memory import ConversationMemoryManager
from qna_bot.config.settings import AppSettings
from qna_bot.database.connection import DatabaseManager
from qna_bot.mcp.client_manager import MCPClientManager
from qna_bot.rag.ingestion import DocumentIngestor
from qna_bot.rag.vector_store import FAISSVectorStoreManager
from qna_bot.services.conversation_service import ConversationService
from qna_bot.services.document_service import DocumentService
from qna_bot.services.mcp_service import MCPService
from qna_bot.services.suggestion_service import SuggestionService
from tests.test_rag import DeterministicTestEmbeddings


def test_dynamic_suggestions_adapt_to_mcp_docs_history_and_cache(tmp_path: Path) -> None:
    db_path = tmp_path / "sug.db"
    settings = AppSettings(
        groq_api_key="",
        data_dir=tmp_path,
        vector_store_dir=tmp_path / "vs",
        sqlite_db_path=db_path,
    )
    db = DatabaseManager(db_path=db_path)
    mem = ConversationMemoryManager(db_path=db_path)
    store = FAISSVectorStoreManager(
        base_dir=settings.vector_store_dir,
        embeddings=DeterministicTestEmbeddings(),
    )
    mcp_mgr = MCPClientManager(db_manager=db)
    mcp_svc = MCPService(client_manager=mcp_mgr, db_manager=db, settings=settings)
    doc_svc = DocumentService(settings=settings, vector_store=store)
    conv_svc = ConversationService(mem)

    try:
        svc = SuggestionService(
            settings=settings,
            mcp_service=mcp_svc,
            document_service=doc_svc,
            conversation_service=conv_svc,
        )

        # Scenario A: No MCP, no docs, no history -> clean fallback suggestions
        cache: dict = {}
        s_a = svc.get_suggestions("alice", current_chat_id="c1", cache_store=cache)
        assert len(s_a) == 4
        assert not any("NeuralVerge" in item.prompt for item in s_a)

        # Caching works across repeated reruns without altering signature
        first_sig = cache["_qnagpt_suggestions_cache"]["signature"]
        s_a_cached = svc.get_suggestions("alice", current_chat_id="c1", cache_store=cache)
        assert s_a_cached == s_a
        assert cache["_qnagpt_suggestions_cache"]["signature"] == first_sig

        # Scenario B + D: Connect MCP tools + upload documents -> context signature changes dynamically
        mcp_mgr.tool_registry.register_server_tools(
            user_id="alice",
            server_id="srv_nv",
            server_name="NeuralVerge",
            raw_tools=[
                SimpleNamespace(
                    name="run_theresanaiforthat_companies",
                    description="Discover AI startups and company funding details",
                    input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
                    annotations=SimpleNamespace(readOnlyHint=True, destructiveHint=False),
                )
            ],
        )
        # Simulate connected server status cleanly for SuggestionService inspection
        with MagicMock() as _:
            pass
        mcp_svc.list_statuses = lambda user_id: [  # type: ignore[method-assign]
            SimpleNamespace(name="NeuralVerge", tool_count=1, transport="streamable_http", connected=True)
        ]

        DocumentIngestor(vector_store=store, settings=settings).ingest_bytes(
            b"Quarterly architecture benchmark report.",
            "architecture_report.txt",
            user_id="alice",
        )
        t_prev = conv_svc.create_thread("alice", title="FastAPI Streaming Patterns")
        mem.append_message(t_prev, "alice", {"role": "user", "content": "FastAPI Streaming Patterns"})

        s_rich = svc.get_suggestions("alice", current_chat_id="c_new", cache_store=cache)
        assert cache["_qnagpt_suggestions_cache"]["signature"] != first_sig
        assert len(s_rich) == 4
        combined_text = " ".join(f"{x.label} {x.prompt}" for x in s_rich)
        assert "NeuralVerge" in combined_text
        assert "architecture_report.txt" in combined_text
        assert "FastAPI Streaming Patterns" in combined_text

        # Bob has zero docs/tools (strict user isolation)
        mcp_svc.list_statuses = lambda user_id: []  # type: ignore[method-assign]
        s_bob = svc.get_suggestions("bob", current_chat_id="c_bob", cache_store={})
        assert "architecture_report.txt" not in " ".join(x.prompt for x in s_bob)
    finally:
        mcp_mgr.shutdown()
        mem.close()
        db.close()
