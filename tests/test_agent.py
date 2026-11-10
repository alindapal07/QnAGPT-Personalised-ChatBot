"""Pytest suite for LangGraph routing, query classification, rewriting, tool calling, validation, and E2E RAG."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from qna_bot.agent.graph import build_agent_graph
from qna_bot.agent.memory import ConversationMemoryManager
from qna_bot.agent.nodes import (
    AgentNodeSuite,
    _QueryAnalysisSchema,
    _TaskPlanSchema,
    _ValidationSchema,
)
from qna_bot.agent.routing import (
    route_after_analysis,
    route_after_grading,
    route_after_validation,
)
from qna_bot.config.settings import AppSettings
from qna_bot.database.connection import DatabaseManager
from qna_bot.mcp.client_manager import MCPClientManager
from qna_bot.mcp.models import ToolExecutionResult
from qna_bot.rag.grader import DocumentGrader, _BatchGradeSchema
from qna_bot.rag.ingestion import DocumentIngestor
from qna_bot.rag.retriever import DocumentRetriever
from qna_bot.rag.vector_store import FAISSVectorStoreManager
from qna_bot.services.chat_service import ChatService
from qna_bot.tools.web_research import ResearchEntityRecord, ResearchExtractionSchema
from qna_bot.tools.web_search import DuckDuckGoWebSearcher, ExtractedWebPage, SafeWebPageReader, WebSearchResult, build_open_web_page_tool
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

        # Verify SQLite checkpoint preserves 3-turn history under Alice's scoped namespace only
        graph = build_agent_graph(user_id="alice", memory_manager=memory, settings=settings)
        alice_tid = chat_service._scoped_checkpoint_id("alice", thread_id)
        bob_tid = chat_service._scoped_checkpoint_id("bob", thread_id)
        snapshot = graph.get_state({"configurable": {"thread_id": alice_tid}})
        checkpoint_msgs = snapshot.values["messages"]
        assert len(checkpoint_msgs) == 6
        bob_snapshot = graph.get_state({"configurable": {"thread_id": bob_tid}})
        assert bob_snapshot.values.get("messages", []) == []
    finally:
        memory.close()


def test_agent_vs_response_llm_and_harmony_browser_open_recovery(tmp_path: Path) -> None:
    settings = AppSettings(
        groq_api_key="gsk_test",
        data_dir=tmp_path,
        vector_store_dir=tmp_path / "vs",
        sqlite_db_path=tmp_path / "chatbot.db",
    )
    store = FAISSVectorStoreManager(
        base_dir=settings.vector_store_dir,
        embeddings=DeterministicTestEmbeddings(),
    )
    memory = ConversationMemoryManager(db_path=settings.sqlite_db_path)

    mock_searcher = MagicMock(spec=DuckDuckGoWebSearcher)
    hit = WebSearchResult(
        title="Python 3.13 Features",
        url="https://docs.python.org/3.13/whatsnew/3.13.html",
        snippet="Free-threaded CPython support.",
    )
    mock_searcher.search.return_value = ([hit], "")
    mock_searcher.get_result_by_id.return_value = hit

    fake_llm = MagicMock()
    analysis_mock = MagicMock()
    analysis_mock.invoke.return_value = _QueryAnalysisSchema(query_type="GENERAL")
    fake_llm.with_structured_output.return_value = analysis_mock

    bound_tool_llm = MagicMock()
    # Turn 1: web_search, Turn 2: model emits Harmony browser.open -> remapped to open_web_page!
    # Turn 3: model chooses to respond in text -> Response LLM synthesizes
    bound_tool_llm.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "web_search",
                    "args": {"query": "Python 3.13 features"},
                    "id": "call_ws_1",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "browser.open",
                    "args": {"id": 1, "cursor": 0},
                    "id": "call_open_1",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(content=""),
    ]
    fake_llm.bind_tools.return_value = bound_tool_llm

    # Test A: Normal Chat never binds tools
    fake_llm.invoke.return_value = AIMessage(content="Dependency injection decouples components.")
    try:
        graph = build_agent_graph(
            user_id="alice",
            api_key="gsk_test",
            retriever=DocumentRetriever(vector_store=store, settings=settings),
            grader=DocumentGrader(api_key="gsk_test", settings=settings),
            web_searcher=mock_searcher,
            memory_manager=memory,
            settings=settings,
        )
        with patch("qna_bot.agent.nodes.ChatGroq", return_value=fake_llm):
            out_a = graph.invoke(
                {
                    "messages": [HumanMessage(content="Explain dependency injection simply")],
                    "user_query": "Explain dependency injection simply",
                    "chat_mode": "General Chat",
                    "user_id": "alice",
                    "thread_id": "t_normal",
                },
                config={"configurable": {"thread_id": "t_normal"}},
            )
            assert "decouples" in out_a["answer"]
            fake_llm.bind_tools.assert_not_called()

            # Test B/C + Response LLM recovery from Groq tool_use_failed
            fake_llm.invoke.side_effect = [
                Exception(
                    "Error code: 400 - {'error': {'message': 'Tool choice is none, but model called a tool', 'code': 'tool_use_failed'}}"
                ),
                AIMessage(content="Python 3.13 adds experimental free-threaded mode and JIT highlights."),
            ]
            with patch(
                "qna_bot.tools.web_search.SafeWebPageReader.fetch_text",
                return_value="Title: What's New In Python 3.13\nExperimental free-threaded mode.",
            ) as mock_fetch:
                out_bc = graph.invoke(
                    {
                        "messages": [HumanMessage(content="Search Python 3.13 features and open result 1")],
                        "user_query": "Search Python 3.13 features and open result 1",
                        "chat_mode": "Auto (Agentic)",
                        "user_id": "alice",
                        "thread_id": "t_browse",
                    },
                    config={"configurable": {"thread_id": "t_browse"}},
                )

        assert mock_fetch.called
        assert out_bc["tool_calls_count"] == 2
        assert "free-threaded" in out_bc["answer"].lower()
        assert fake_llm.bind_tools.call_args.kwargs.get("tool_choice") == "auto"

        # Verify Response LLM was never handed raw ToolMessage or AIMessage(tool_calls=...)
        for call in fake_llm.invoke.call_args_list:
            passed_msgs = call.args[0]
            assert not any(isinstance(m, ToolMessage) for m in passed_msgs)
            assert not any(isinstance(m, AIMessage) and getattr(m, "tool_calls", None) for m in passed_msgs)
    finally:
        memory.close()


def test_multi_mcp_server_binding_cap_and_zero_server_agent_fallback(tmp_path: Path) -> None:
    db_path = tmp_path / "multi_mcp.db"
    settings = AppSettings(
        groq_api_key="gsk_test",
        data_dir=tmp_path,
        vector_store_dir=tmp_path / "vs",
        sqlite_db_path=db_path,
    )
    db = DatabaseManager(db_path=db_path)
    memory = ConversationMemoryManager(db_path=db_path)
    mcp_mgr = MCPClientManager(db_manager=db)
    store = FAISSVectorStoreManager(
        base_dir=settings.vector_store_dir,
        embeddings=DeterministicTestEmbeddings(),
    )

    # Test D & E: NeuralVerge (380 tools) + GitHub (2 tools) with zero name collisions
    nv_tools = [
        SimpleNamespace(
            name=f"run_actor_{idx}" if idx else "run_theresanaiforthat_companies",
            description=f"Run dataset actor {idx}",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            annotations=SimpleNamespace(readOnlyHint=True, destructiveHint=False),
        )
        for idx in range(380)
    ]
    gh_tools = [
        SimpleNamespace(
            name="search_repositories",
            description="Search GitHub repositories",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            annotations=SimpleNamespace(readOnlyHint=True, destructiveHint=False),
        )
    ]
    mcp_mgr.tool_registry.register_server_tools("alice", "srv_nv", "NeuralVerge", nv_tools)
    mcp_mgr.tool_registry.register_server_tools("alice", "srv_gh", "GitHub", gh_tools)

    fake_llm = MagicMock()
    def _structured(cls):
        m = MagicMock()
        if cls.__name__ == "_QueryAnalysisSchema":
            m.invoke.return_value = _QueryAnalysisSchema(query_type="AGENT_TASK", web_search_required=True)
        elif cls.__name__ == "_TaskPlanSchema":
            m.invoke.return_value = _TaskPlanSchema(steps=["Query companies", "Summarize output"])
        return m

    fake_llm.with_structured_output.side_effect = _structured
    bound_llm = MagicMock()
    bound_llm.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "neuralverge__run_theresanaiforthat_companies",
                    "args": {"query": "AI startups"},
                    "id": "call_nv_1",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(content=""),
    ]
    fake_llm.bind_tools.return_value = bound_llm
    fake_llm.invoke.return_value = AIMessage(content="Completed NeuralVerge dataset query cleanly.")

    try:
        graph = build_agent_graph(
            user_id="alice",
            api_key="gsk_test",
            retriever=DocumentRetriever(vector_store=store, settings=settings),
            grader=DocumentGrader(api_key="gsk_test", settings=settings),
            memory_manager=memory,
            mcp_manager=mcp_mgr,
            settings=settings,
        )
        with (
            patch("qna_bot.agent.nodes.ChatGroq", return_value=fake_llm),
            patch.object(
                mcp_mgr,
                "execute_tool",
                return_value=ToolExecutionResult(
                    server_id="srv_nv",
                    server_name="NeuralVerge",
                    tool_name="run_theresanaiforthat_companies",
                    success=True,
                    content='[{"company": "ExampleAI"}]',
                    duration_ms=42.0,
                ),
            ),
        ):
            out = graph.invoke(
                {
                    "messages": [HumanMessage(content="Get AI companies from NeuralVerge")],
                    "user_query": "Get AI companies from NeuralVerge",
                    "chat_mode": "Auto (Agentic)",
                    "user_id": "alice",
                    "thread_id": "t_nv",
                },
                config={"configurable": {"thread_id": "t_nv"}},
            )
        assert "ExampleAI" in str(out["tool_executions"]) or "Completed NeuralVerge" in out["answer"]
        bound_list = fake_llm.bind_tools.call_args.args[0]
        # 35 MCP cap + 4 built-ins (retrieve, summarize, web_search, open_web_page)
        assert len(bound_list) <= 39
        assert "open_web_page" in {t.name for t in bound_list}
        assert fake_llm.bind_tools.call_args.kwargs.get("tool_choice") == "auto"

        # Test F: No Connected MCP Servers still executes built-in search+open in AGENT_TASK
        empty_db = DatabaseManager(db_path=tmp_path / "empty_mcp.db")
        empty_mcp = MCPClientManager(db_manager=empty_db)
        mock_searcher = MagicMock(spec=DuckDuckGoWebSearcher)
        hit = WebSearchResult(title="Doc", url="https://example.com/doc", snippet="Summary snippet")
        mock_searcher.search.return_value = ([hit], "")
        mock_searcher.get_result_by_id.return_value = hit

        bound_llm.invoke.side_effect = [
            AIMessage(
                content="",
                tool_calls=[{"name": "web_search", "args": {"query": "Doc"}, "id": "c1", "type": "tool_call"}],
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": "open_web_page", "args": {"result_id": 1}, "id": "c2", "type": "tool_call"}],
            ),
            AIMessage(content=""),
        ]
        fake_llm.invoke.return_value = AIMessage(content="Read https://example.com/doc via built-in browser.")
        empty_graph = build_agent_graph(
            user_id="bob",
            api_key="gsk_test",
            retriever=DocumentRetriever(vector_store=store, settings=settings),
            grader=DocumentGrader(api_key="gsk_test", settings=settings),
            web_searcher=mock_searcher,
            memory_manager=memory,
            mcp_manager=empty_mcp,
            settings=settings,
        )
        with (
            patch("qna_bot.agent.nodes.ChatGroq", return_value=fake_llm),
            patch("qna_bot.tools.web_search.SafeWebPageReader.fetch_text", return_value="Full document text"),
        ):
            out_f = empty_graph.invoke(
                {
                    "messages": [HumanMessage(content="Search for Doc and open the first result")],
                    "user_query": "Search for Doc and open the first result",
                    "chat_mode": "Auto (Agentic)",
                    "user_id": "bob",
                    "thread_id": "t_empty_mcp",
                },
                config={"configurable": {"thread_id": "t_empty_mcp"}},
            )
        assert "example.com/doc" in out_f["answer"]
        assert len(out_f["tool_executions"]) == 2
        empty_db.close()
    finally:
        memory.close()
        db.close()


def test_deep_web_research_pipeline_opens_pages_extracts_verifies_and_handles_failed_pages(
    tmp_path: Path,
) -> None:
    settings = AppSettings(
        groq_api_key="gsk_test",
        data_dir=tmp_path,
        vector_store_dir=tmp_path / "vs",
        sqlite_db_path=tmp_path / "research.db",
    )
    store = FAISSVectorStoreManager(
        base_dir=settings.vector_store_dir,
        embeddings=DeterministicTestEmbeddings(),
    )
    memory = ConversationMemoryManager(db_path=settings.sqlite_db_path)

    mock_searcher = MagicMock(spec=DuckDuckGoWebSearcher)
    mock_searcher.search.return_value = (
        [
            WebSearchResult(
                title="UpForge Blocked Post",
                url="https://upforge.example.com/startups",
                snippet="Blog post link only.",
            ),
            WebSearchResult(
                title="Seedtable India AI",
                url="https://seedtable.example.org/ai-india",
                snippet="List of top Indian AI startups.",
            ),
            WebSearchResult(
                title="Funding Tracker India",
                url="https://tracker.example.net/india-ai",
                snippet="Startup tracker.",
            ),
        ],
        "",
    )

    def _fake_extract_page(url: str) -> ExtractedWebPage:
        if "upforge" in url:
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
        if "seedtable" in url:
            return ExtractedWebPage(
                requested_url=url,
                final_url=url,
                canonical_url=url,
                title="Seedtable AI India",
                description="",
                content=(
                    "| Company | Founders | Website | Funding |\n"
                    "| --- | --- | --- | --- |\n"
                    "| Sarvam AI | Vivek Raghavan, Pratyush Kumar | https://sarvam.ai | $41M |"
                ),
                status_code=200,
                success=True,
                char_count=140,
            )
        return ExtractedWebPage(
            requested_url=url,
            final_url=url,
            canonical_url=url,
            title="Funding Tracker India",
            description="",
            content="Sarvam AI (Vivek Raghavan) raised $41M. Krutrim (Bhavish Aggarwal) website https://olakrutrim.com raised $50M.",
            status_code=200,
            success=True,
            char_count=110,
        )

    fake_llm = MagicMock()

    def _structured(schema_cls):
        m = MagicMock()
        if schema_cls.__name__ == "_QueryAnalysisSchema":
            m.invoke.return_value = _QueryAnalysisSchema(
                query_type="WEB_SEARCH",
                web_search_required=True,
            )
        elif schema_cls.__name__ == "ResearchExtractionSchema":
            m.invoke.return_value = ResearchExtractionSchema(
                companies=[
                    ResearchEntityRecord(
                        name="Sarvam AI",
                        founders=["Vivek Raghavan", "Pratyush Kumar"],
                        website="https://sarvam.ai",
                        funding_amount="$41M",
                        source_urls=[
                            "https://seedtable.example.org/ai-india",
                            "https://tracker.example.net/india-ai",
                        ],
                    ),
                    ResearchEntityRecord(
                        name="Krutrim",
                        founders=["Bhavish Aggarwal"],
                        website="https://olakrutrim.com",
                        funding_amount="$50M",
                        source_urls=["https://tracker.example.net/india-ai"],
                    ),
                ]
            )
        return m

    fake_llm.with_structured_output.side_effect = _structured
    fake_llm.invoke.return_value = AIMessage(
        content=(
            "| Company | Founders | Website | Funding |\n"
            "| --- | --- | --- | --- |\n"
            "| Sarvam AI | Vivek Raghavan, Pratyush Kumar | https://sarvam.ai | $41M |\n"
            "| Krutrim | Bhavish Aggarwal | https://olakrutrim.com | $50M |"
        )
    )

    try:
        chat_svc = ChatService(
            settings=settings,
            memory_manager=memory,
            vector_store=store,
            web_searcher=mock_searcher,
        )
        thread_id = chat_svc.create_thread("alice", "Startup Research")
        with (
            patch("qna_bot.agent.nodes.ChatGroq", return_value=fake_llm),
            patch("qna_bot.tools.web_search.SafeWebPageReader.extract_page", side_effect=_fake_extract_page),
        ):
            res = chat_svc.run_agentic_turn(
                user_query="Find five AI startups in India with founders, websites, and funding details.",
                thread_id=thread_id,
                user_id="alice",
                chat_mode="Auto (Agentic)",
                model_name="openai/gpt-oss-20b",
                temperature=0.0,
                max_tokens=1024,
                system_prompt="",
            )

        assert "Sarvam AI" in res.answer
        assert "$41M" in res.answer
        # Verify opened page content and structured verification were passed to generate_answer
        passed_msgs = fake_llm.invoke.call_args.args[0]
        user_payload = str(passed_msgs[-1].content)
        assert "EXTRACTED WEBPAGE CONTENT" in user_payload
        assert "CROSS-VERIFIED STRUCTURED EXTRACTION" in user_payload
        assert "Vivek Raghavan" in user_payload

        # Verify UI citations distinguish accessible vs inaccessible sources cleanly
        src_statuses = {s["url"]: s.get("status") for s in res.web_sources}
        assert src_statuses["https://upforge.example.com/startups"] == "inaccessible"
        assert src_statuses["https://seedtable.example.org/ai-india"] == "read"
        assert src_statuses["https://tracker.example.net/india-ai"] == "read"
        assert len(res.tool_executions) == 3
    finally:
        memory.close()


