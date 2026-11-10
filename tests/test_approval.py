"""Pytest suite for backend HITL approval binding, LangGraph interrupt/resume, and rejection handling."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from qna_bot.agent.approval import ApprovalManager, compute_arguments_hash
from qna_bot.agent.graph import build_agent_graph
from qna_bot.agent.memory import ConversationMemoryManager
from qna_bot.agent.nodes import _QueryAnalysisSchema, _TaskPlanSchema
from qna_bot.config.settings import AppSettings
from qna_bot.database.connection import DatabaseManager
from qna_bot.mcp.client_manager import MCPClientManager
from qna_bot.mcp.models import ToolExecutionResult
from qna_bot.rag.grader import DocumentGrader
from qna_bot.rag.retriever import DocumentRetriever
from qna_bot.rag.vector_store import FAISSVectorStoreManager
from tests.test_rag import DeterministicTestEmbeddings


def test_approval_manager_binds_user_thread_and_exact_arguments(tmp_path: Path) -> None:
    db = DatabaseManager(db_path=tmp_path / "appr.db")
    try:
        mgr = ApprovalManager(db_manager=db)
        rec = mgr.create_pending_approval(
            user_id="alice",
            thread_id="thread_1",
            server_id="srv_fs",
            tool_name="write_file",
            arguments={"path": "notes.md", "token": "secret_123"},
            category="MODIFICATION",
            reason="Create notes.md",
        )
        assert "secret_123" not in rec.masked_arguments
        assert rec.arguments_hash == compute_arguments_hash({"path": "notes.md", "token": "secret_123"})

        # Bob cannot approve Alice's request
        assert mgr.decide(rec.approval_id, user_id="bob", approved=True) is None

        # Pending cannot be consumed before approval
        assert not mgr.verify_and_consume(
            rec.approval_id, "alice", "thread_1", "srv_fs", "write_file", {"path": "notes.md", "token": "secret_123"}
        )

        mgr.decide(rec.approval_id, user_id="alice", approved=True)

        # Mutated arguments must fail verification
        assert not mgr.verify_and_consume(
            rec.approval_id, "alice", "thread_1", "srv_fs", "write_file", {"path": "evil.md", "token": "secret_123"}
        )

        # Exact match succeeds once, then cannot be replayed
        assert mgr.verify_and_consume(
            rec.approval_id, "alice", "thread_1", "srv_fs", "write_file", {"path": "notes.md", "token": "secret_123"}
        )
        assert not mgr.verify_and_consume(
            rec.approval_id, "alice", "thread_1", "srv_fs", "write_file", {"path": "notes.md", "token": "secret_123"}
        )
    finally:
        db.close()


def _setup_mcp_graph_env(tmp_path: Path):
    db_path = tmp_path / "hitl.db"
    settings = AppSettings(
        groq_api_key="gsk_test",
        data_dir=tmp_path,
        vector_store_dir=tmp_path / "vs",
        sqlite_db_path=db_path,
    )
    db = DatabaseManager(db_path=db_path)
    memory = ConversationMemoryManager(db_path=db_path)
    mcp_mgr = MCPClientManager(db_manager=db)
    mcp_mgr.tool_registry.register_server_tools(
        user_id="alice",
        server_id="srv_fs",
        server_name="Filesystem",
        raw_tools=[
            SimpleNamespace(
                name="write_file",
                description="Write text to a file",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"],
                },
                annotations=SimpleNamespace(readOnlyHint=False, destructiveHint=False),
            )
        ],
    )
    store = FAISSVectorStoreManager(
        base_dir=settings.vector_store_dir,
        embeddings=DeterministicTestEmbeddings(),
    )
    approval_mgr = ApprovalManager(db_manager=db)
    graph = build_agent_graph(
        user_id="alice",
        api_key="gsk_test",
        retriever=DocumentRetriever(vector_store=store, settings=settings),
        grader=DocumentGrader(api_key="gsk_test", settings=settings),
        memory_manager=memory,
        mcp_manager=mcp_mgr,
        approval_manager=approval_mgr,
        settings=settings,
    )
    return db, memory, mcp_mgr, approval_mgr, graph


def test_langgraph_hitl_approval_interrupt_and_resume_executes_tool(tmp_path: Path) -> None:
    db, memory, mcp_mgr, approval_mgr, graph = _setup_mcp_graph_env(tmp_path)

    def structured_router(schema_cls):
        m = MagicMock()
        if schema_cls.__name__ == "_QueryAnalysisSchema":
            m.invoke.return_value = _QueryAnalysisSchema(query_type="AGENT_TASK")
        elif schema_cls.__name__ == "_TaskPlanSchema":
            m.invoke.return_value = _TaskPlanSchema(steps=["Write report.md via Filesystem"])
        return m

    fake_llm = MagicMock()
    fake_llm.with_structured_output.side_effect = structured_router

    bound_llm = MagicMock()
    bound_llm.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "filesystem__write_file",
                    "args": {"path": "report.md", "content": "Summary"},
                    "id": "call_write_1",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(content="Saved report.md successfully."),
    ]
    fake_llm.bind_tools.return_value = bound_llm
    fake_llm.invoke.return_value = AIMessage(content="Report saved to report.md.")

    try:
        with (
            patch("qna_bot.agent.nodes.ChatGroq", return_value=fake_llm),
            patch.object(
                mcp_mgr,
                "execute_tool",
                return_value=ToolExecutionResult(
                    server_id="srv_fs",
                    server_name="Filesystem",
                    tool_name="write_file",
                    success=True,
                    content="Wrote 7 bytes to report.md",
                    duration_ms=12.0,
                ),
            ) as mock_exec,
        ):
            cfg = {"configurable": {"thread_id": "t_hitl_approve"}}
            paused = graph.invoke(
                {
                    "messages": [HumanMessage(content="Save a summary to report.md")],
                    "user_query": "Save a summary to report.md",
                    "chat_mode": "Auto (Agentic)",
                    "user_id": "alice",
                    "thread_id": "t_hitl_approve",
                },
                cfg,
            )
            assert "__interrupt__" in paused
            interrupt_payload = paused["__interrupt__"][0].value
            approval_id = interrupt_payload["approval_id"]
            assert interrupt_payload["tool_name"] == "write_file"
            mock_exec.assert_not_called()

            # Approve in backend and resume graph
            approval_mgr.decide(approval_id, user_id="alice", approved=True)
            resumed = graph.invoke(
                Command(resume={"approved": True, "approval_id": approval_id}),
                cfg,
            )
            mock_exec.assert_called_once()
            assert "report.md" in resumed["answer"]
    finally:
        memory.close()
        db.close()


def test_langgraph_hitl_rejection_blocks_execution_cleanly(tmp_path: Path) -> None:
    db, memory, mcp_mgr, approval_mgr, graph = _setup_mcp_graph_env(tmp_path)

    def structured_router(schema_cls):
        m = MagicMock()
        if schema_cls.__name__ == "_QueryAnalysisSchema":
            m.invoke.return_value = _QueryAnalysisSchema(query_type="AGENT_TASK")
        elif schema_cls.__name__ == "_TaskPlanSchema":
            m.invoke.return_value = _TaskPlanSchema(steps=["Write report.md"])
        return m

    fake_llm = MagicMock()
    fake_llm.with_structured_output.side_effect = structured_router
    bound_llm = MagicMock()
    bound_llm.invoke.return_value = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "filesystem__write_file",
                "args": {"path": "report.md", "content": "Summary"},
                "id": "call_write_reject",
                "type": "tool_call",
            }
        ],
    )
    fake_llm.bind_tools.return_value = bound_llm
    fake_llm.invoke.return_value = AIMessage(content="The write action was rejected and not executed.")

    try:
        with (
            patch("qna_bot.agent.nodes.ChatGroq", return_value=fake_llm),
            patch.object(mcp_mgr, "execute_tool") as mock_exec,
        ):
            cfg = {"configurable": {"thread_id": "t_hitl_reject"}}
            paused = graph.invoke(
                {
                    "messages": [HumanMessage(content="Save report.md")],
                    "user_query": "Save report.md",
                    "chat_mode": "Auto (Agentic)",
                    "user_id": "alice",
                    "thread_id": "t_hitl_reject",
                },
                cfg,
            )
            approval_id = paused["__interrupt__"][0].value["approval_id"]
            approval_mgr.decide(approval_id, user_id="alice", approved=False)
            resumed = graph.invoke(
                Command(resume={"approved": False, "approval_id": approval_id}),
                cfg,
            )
            mock_exec.assert_not_called()
            assert "rejected" in resumed["answer"].lower()
            assert resumed["tool_executions"][-1]["status"] == "rejected"
    finally:
        memory.close()
        db.close()
