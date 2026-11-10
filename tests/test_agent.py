"""Pytest suite for LangGraph routing, query classification, rewriting, tool calling, validation, and E2E RAG."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from qna_bot.agent.graph import build_agent_graph
from qna_bot.agent.memory import ConversationMemoryManager
from qna_bot.agent.nodes import AgentNodeSuite, _QueryAnalysisSchema, _ValidationSchema
from qna_bot.agent.routing import (
    route_after_analysis,
    route_after_grading,
    route_after_validation,
)
from qna_bot.config.settings import AppSettings
from qna_bot.rag.grader import DocumentGrader, _BatchGradeSchema
from qna_bot.rag.ingestion import DocumentIngestor
from qna_bot.rag.retriever import DocumentRetriever
from qna_bot.rag.vector_store import FAISSVectorStoreManager
from qna_bot.services.chat_service import ChatService
from qna_bot.tools.web_search import DuckDuckGoWebSearcher, WebSearchResult
from tests.test_rag import DeterministicTestEmbeddings, _make_pdf_bytes


def test_deterministic_graph_routing_and_loop_bounds() -> None:
    assert route_after_analysis({"chat_mode": "General Chat", "query_type": "DOCUMENT"}) == "direct_answer"
    assert route_after_analysis({"chat_mode": "Document Chat", "query_type": "GENERAL"}) == "rewrite_query"
    assert (
        route_after_analysis(
            {"chat_mode": "Auto (Agentic)", "query_type": "DOCUMENT", "document_search_required": True}
        )
        == "rewrite_query"
    )
    assert (
        route_after_analysis(
            {"chat_mode": "Auto (Agentic)", "query_type": "WEB_SEARCH", "web_search_required": True}
        )
        == "web_search"
    )

    assert route_after_grading({"relevant_documents": [{"content": "hit"}]}) == "generate_answer"
    assert route_after_grading({"relevant_documents": [], "retry_count": 0, "chat_mode": "Auto (Agentic)"}) == "rewrite_query"
    assert route_after_grading({"relevant_documents": [], "retry_count": 1, "chat_mode": "Auto (Agentic)"}) == "web_search"
    assert route_after_grading({"relevant_documents": [], "retry_count": 2, "chat_mode": "Document Chat"}) == "generate_answer"

    assert route_after_validation({"validation_passed": False, "validation_retries": 0}) == "generate_answer"
    assert route_after_validation({"validation_passed": False, "validation_retries": 1}) == "end"
    assert route_after_validation({"validation_passed": True, "validation_retries": 0}) == "end"


def test_query_classification_and_rewriting_with_mocked_groq(tmp_path: Path) -> None:
    settings = AppSettings(groq_api_key="gsk_test", data_dir=tmp_path, vector_store_dir=tmp_path / "vs")
    store = FAISSVectorStoreManager(
        base_dir=settings.vector_store_dir,
        embeddings=DeterministicTestEmbeddings(),
    )
    DocumentIngestor(vector_store=store, settings=settings).ingest_bytes(
        b"Neural networks learn layered representations.",
        "nn.txt",
        user_id="alice",
    )
    retriever = DocumentRetriever(vector_store=store, settings=settings)
    nodes = AgentNodeSuite(
        retriever=retriever,
        grader=DocumentGrader(api_key="gsk_test", settings=settings),
        web_searcher=DuckDuckGoWebSearcher(),
        tools=[],
        api_key="gsk_test",
        settings=settings,
    )

    mock_structured = MagicMock()
    mock_structured.invoke.return_value = _QueryAnalysisSchema(
        query_type="FOLLOW_UP",
        document_search_required=True,
        web_search_required=False,
    )
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    mock_llm.invoke.return_value = AIMessage(content="What are the disadvantages of neural networks?")

    with patch.object(nodes, "_make_llm", return_value=mock_llm):
        analysis = nodes.analyze_query(
            {
                "user_query": "What are its disadvantages?",
                "chat_mode": "Auto (Agentic)",
                "user_id": "alice",
                "messages": [
                    HumanMessage(content="Explain neural networks."),
                    AIMessage(content="Neural networks are layered models."),
                ],
            }
        )
        assert analysis["query_type"] == "FOLLOW_UP"
        assert analysis["document_search_required"] is True

        rewritten = nodes.rewrite_query(
            {
                "user_query": "What are its disadvantages?",
                "messages": [
                    HumanMessage(content="Explain neural networks."),
                    AIMessage(content="Neural networks are layered models."),
                ],
            }
        )
        assert rewritten["rewritten_query"] == "What are the disadvantages of neural networks?"


def test_tool_calling_integration_via_langgraph_toolnode(tmp_path: Path) -> None:
    settings = AppSettings(groq_api_key="gsk_test", data_dir=tmp_path, vector_store_dir=tmp_path / "vs", sqlite_db_path=tmp_path / "db.sqlite")
    store = FAISSVectorStoreManager(
        base_dir=settings.vector_store_dir,
        embeddings=DeterministicTestEmbeddings(),
    )
    DocumentIngestor(vector_store=store, settings=settings).ingest_bytes(
        b"Supervised learning uses labeled datasets for training.",
        "ml.txt",
        user_id="alice",
    )
    memory = ConversationMemoryManager(db_path=settings.sqlite_db_path)

    # First call: classify GENERAL -> direct_answer triggers tool call -> ToolNode runs -> direct_answer summarizes
    fake_llm = MagicMock()
    structured_analysis = MagicMock()
    structured_analysis.invoke.return_value = _QueryAnalysisSchema(
        query_type="GENERAL",
        document_search_required=False,
        web_search_required=False,
    )
    fake_llm.with_structured_output.return_value = structured_analysis

    bound_tool_llm = MagicMock()
    bound_tool_llm.invoke.return_value = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "retrieve_uploaded_documents",
                "args": {"query": "supervised learning", "top_k": 2},
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )
    fake_llm.bind_tools.return_value = bound_tool_llm
    fake_llm.invoke.return_value = AIMessage(
        content="Based on ml.txt, supervised learning uses labeled datasets."
    )

    try:
        graph = build_agent_graph(
            user_id="alice",
            api_key="gsk_test",
            retriever=DocumentRetriever(vector_store=store, settings=settings),
            grader=DocumentGrader(api_key="gsk_test", settings=settings),
            web_searcher=DuckDuckGoWebSearcher(),
            memory_manager=memory,
            settings=settings,
        )
        with patch("qna_bot.agent.nodes.ChatGroq", return_value=fake_llm):
            out = graph.invoke(
                {
                    "messages": [HumanMessage(content="Check my notes quickly.")],
                    "user_query": "Check my notes quickly.",
                    "chat_mode": "Auto (Agentic)",
                    "user_id": "alice",
                    "thread_id": "t_tools",
                },
                config={"configurable": {"thread_id": "t_tools"}},
            )
        assert "supervised learning" in out["answer"].lower()
        assert out["tool_calls_count"] == 1
    finally:
        memory.close()


def test_end_to_end_agentic_rag_and_web_search_workflow(tmp_path: Path) -> None:
    settings = AppSettings(
        groq_api_key="gsk_test",
        data_dir=tmp_path,
        vector_store_dir=tmp_path / "vs",
        sqlite_db_path=tmp_path / "chatbot.db",
        retrieval_score_threshold=0.1,
    )
    store = FAISSVectorStoreManager(
        base_dir=settings.vector_store_dir,
        embeddings=DeterministicTestEmbeddings(),
    )
    DocumentIngestor(vector_store=store, settings=settings).ingest_bytes(
        _make_pdf_bytes("Supervised learning uses labeled datasets to train predictive models."),
        "ML_Research.pdf",
        user_id="alice",
    )

    mock_searcher = MagicMock(spec=DuckDuckGoWebSearcher)
    mock_searcher.search.return_value = (
        [
            WebSearchResult(
                title="Latest AI Update",
                url="https://example.com/ai-update",
                snippet="Recent developments in open-weight models.",
            )
        ],
        "",
    )

    memory = ConversationMemoryManager(db_path=settings.sqlite_db_path)
    chat_service = ChatService(
        settings=settings,
        memory_manager=memory,
        vector_store=store,
        web_searcher=mock_searcher,
    )

    # Configure mocked ChatGroq responses across analyze, rewrite, grade, generate, validate
    analysis_mock = MagicMock()
    analysis_mock.invoke.side_effect = [
        _QueryAnalysisSchema(query_type="DOCUMENT", document_search_required=True),
        _QueryAnalysisSchema(query_type="WEB_SEARCH", web_search_required=True),
        _QueryAnalysisSchema(query_type="FOLLOW_UP", document_search_required=True),
    ]
    grade_mock = MagicMock()
    grade_mock.invoke.return_value = _BatchGradeSchema(
        overall_relevance="RELEVANT",
        relevant_indices=[0],
        reason="Directly answers query.",
    )
    validate_mock = MagicMock()
    validate_mock.invoke.return_value = _ValidationSchema(is_valid=True, feedback="")

    def structured_router(schema_cls):
        name = schema_cls.__name__
        if name == "_QueryAnalysisSchema":
            return analysis_mock
        if name == "_BatchGradeSchema":
            return grade_mock
        return validate_mock

    fake_llm = MagicMock()
    fake_llm.with_structured_output.side_effect = structured_router
    fake_llm.invoke.side_effect = [
        # Turn 1: rewrite -> generate
        AIMessage(content="supervised learning labeled datasets"),
        AIMessage(content="According to ML_Research.pdf (Page 1), supervised learning uses labeled datasets."),
        # Turn 2: web generate
        AIMessage(content="Recent AI developments include new open-weight releases."),
        # Turn 3: follow-up rewrite -> generate
        AIMessage(content="supervised learning predictive models advantages"),
        AIMessage(content="It trains predictive models directly from labeled examples."),
    ]

    try:
        thread_id = chat_service.create_thread(user_id="alice")
        activities: list[str] = []

        with (
            patch("qna_bot.agent.nodes.ChatGroq", return_value=fake_llm),
            patch("qna_bot.rag.grader.ChatGroq", return_value=fake_llm),
        ):
            # Turn 1: Document RAG
            res1 = chat_service.run_agentic_turn(
                user_query="What does ML_Research say about supervised learning?",
                thread_id=thread_id,
                user_id="alice",
                chat_mode="Auto (Agentic)",
                model_name="openai/gpt-oss-20b",
                temperature=0.7,
                max_tokens=1024,
                system_prompt="",
                on_activity=activities.append,
            )
            assert "ML_Research.pdf" in res1.answer
            assert len(res1.sources) == 1
            assert res1.sources[0]["source"] == "ML_Research.pdf"
            assert res1.sources[0]["page"] == 1
            assert res1.web_sources == []
            assert "Searching uploaded documents..." in activities

            # Turn 2: Web search (kept separate from document sources)
            res2 = chat_service.run_agentic_turn(
                user_query="What are the latest AI developments?",
                thread_id=thread_id,
                user_id="alice",
                chat_mode="Auto (Agentic)",
                model_name="openai/gpt-oss-20b",
                temperature=0.7,
                max_tokens=1024,
                system_prompt="",
            )
            assert res2.sources == []
            assert len(res2.web_sources) == 1
            assert res2.web_sources[0]["url"] == "https://example.com/ai-update"

            # Turn 3: Follow-up on same thread_id preserves checkpoint history
            res3 = chat_service.run_agentic_turn(
                user_query="How does it train models?",
                thread_id=thread_id,
                user_id="alice",
                chat_mode="Auto (Agentic)",
                model_name="openai/gpt-oss-20b",
                temperature=0.7,
                max_tokens=1024,
                system_prompt="",
            )
            assert "predictive models" in res3.answer

        # Verify SQLite checkpoint actually contains all 3 turns
        graph = build_agent_graph(user_id="alice", memory_manager=memory, settings=settings)
        snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
        checkpoint_msgs = snapshot.values["messages"]
        assert len(checkpoint_msgs) == 6
    finally:
        memory.close()
