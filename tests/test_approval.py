"""Pytest suite for backend HITL approval binding, LangGraph interrupt/resume, and rejection handling."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch
import threading
from concurrent.futures import ThreadPoolExecutor
import sqlite3

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from qna_bot.agent.approval import ApprovalManager, compute_arguments_hash
from qna_bot.agent.graph import build_agent_graph
from qna_bot.agent.memory import ConversationMemoryManager
from qna_bot.agent.nodes import _QueryAnalysisSchema, _TaskPlanSchema
from qna_bot.config.settings import AppSettings
from qna_bot.database.connection import DatabaseManager
from qna_bot.mcp_client.client_manager import MCPClientManager
from qna_bot.mcp_client.models import ToolExecutionResult
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


# ============================================================================
# PHASE 7.1 Regression Tests: Atomic MCP Approval Consumption (Tests A - H)
# ============================================================================


def test_atomic_approval_consumption_normal(tmp_path: Path) -> None:
    """Test A: Normal consumption: returns True, status becomes CONSUMED."""
    db = DatabaseManager(db_path=tmp_path / "test_a.db")
    try:
        mgr = ApprovalManager(db_manager=db)
        rec = mgr.create_pending_approval(
            user_id="user_a",
            thread_id="th_a",
            server_id="srv_a",
            tool_name="tool_a",
            arguments={"param": "val1"},
            category="READ",
            reason="Test A",
        )
        decided = mgr.decide(rec.approval_id, user_id="user_a", approved=True)
        assert decided is not None
        assert decided.status == "APPROVED"

        # Verify and consume
        consumed = mgr.verify_and_consume(
            approval_id=rec.approval_id,
            user_id="user_a",
            thread_id="th_a",
            server_id="srv_a",
            tool_name="tool_a",
            arguments={"param": "val1"},
        )
        assert consumed is True

        # Check DB status
        final_rec = mgr.repo.get(rec.approval_id, "user_a")
        assert final_rec is not None
        assert final_rec.status == "CONSUMED"
    finally:
        db.close()


def test_atomic_approval_consumption_second_attempt_rejected(tmp_path: Path) -> None:
    """Test B: Re-consumption rejected: second verify_and_consume returns False."""
    db = DatabaseManager(db_path=tmp_path / "test_b.db")
    try:
        mgr = ApprovalManager(db_manager=db)
        rec = mgr.create_pending_approval(
            user_id="user_b",
            thread_id="th_b",
            server_id="srv_b",
            tool_name="tool_b",
            arguments={"data": 123},
            category="MODIFICATION",
            reason="Test B",
        )
        mgr.decide(rec.approval_id, user_id="user_b", approved=True)

        # First consumption succeeds
        assert mgr.verify_and_consume(
            rec.approval_id, "user_b", "th_b", "srv_b", "tool_b", {"data": 123}
        ) is True

        # Second consumption fails
        assert mgr.verify_and_consume(
            rec.approval_id, "user_b", "th_b", "srv_b", "tool_b", {"data": 123}
        ) is False

        # Status remains CONSUMED
        final_rec = mgr.repo.get(rec.approval_id, "user_b")
        assert final_rec is not None
        assert final_rec.status == "CONSUMED"
    finally:
        db.close()


def test_atomic_approval_consumption_concurrent_race(tmp_path: Path) -> None:
    """Test C: Concurrent consumption race: exactly one returns True, one returns False."""
    db = DatabaseManager(db_path=tmp_path / "test_c.db")
    try:
        mgr = ApprovalManager(db_manager=db)
        rec = mgr.create_pending_approval(
            user_id="user_c",
            thread_id="th_c",
            server_id="srv_c",
            tool_name="tool_c",
            arguments={"item": "target"},
            category="MODIFICATION",
            reason="Test C Race",
        )
        mgr.decide(rec.approval_id, user_id="user_c", approved=True)

        barrier = threading.Barrier(2)

        def worker() -> bool:
            # Sync start to maximize concurrency race condition
            barrier.wait()
            return mgr.verify_and_consume(
                rec.approval_id, "user_c", "th_c", "srv_c", "tool_c", {"item": "target"}
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            fut1 = executor.submit(worker)
            fut2 = executor.submit(worker)
            res1 = fut1.result()
            res2 = fut2.result()

        # Exactly one succeeded and one failed
        results = [res1, res2]
        assert results.count(True) == 1
        assert results.count(False) == 1

        # Final record status in DB is CONSUMED
        final_rec = mgr.repo.get(rec.approval_id, "user_c")
        assert final_rec is not None
        assert final_rec.status == "CONSUMED"
    finally:
        db.close()


def test_atomic_approval_consumption_wrong_arguments(tmp_path: Path) -> None:
    """Test D: Modified arguments rejected: returns False, record remains APPROVED."""
    db = DatabaseManager(db_path=tmp_path / "test_d.db")
    try:
        mgr = ApprovalManager(db_manager=db)
        rec = mgr.create_pending_approval(
            user_id="user_d",
            thread_id="th_d",
            server_id="srv_d",
            tool_name="tool_d",
            arguments={"action": "read"},
            category="READ",
            reason="Test D Args",
        )
        mgr.decide(rec.approval_id, user_id="user_d", approved=True)

        # Attempt consumption with altered arguments
        consumed = mgr.verify_and_consume(
            rec.approval_id, "user_d", "th_d", "srv_d", "tool_d", {"action": "write"}
        )
        assert consumed is False

        # Status must remain APPROVED (unconsumed)
        current_rec = mgr.repo.get(rec.approval_id, "user_d")
        assert current_rec is not None
        assert current_rec.status == "APPROVED"

        # Subsequent consumption with correct arguments still works
        valid_consumed = mgr.verify_and_consume(
            rec.approval_id, "user_d", "th_d", "srv_d", "tool_d", {"action": "read"}
        )
        assert valid_consumed is True
    finally:
        db.close()


def test_atomic_approval_consumption_wrong_user(tmp_path: Path) -> None:
    """Test E: Wrong user rejected: returns False, record remains APPROVED."""
    db = DatabaseManager(db_path=tmp_path / "test_e.db")
    try:
        mgr = ApprovalManager(db_manager=db)
        rec = mgr.create_pending_approval(
            user_id="alice",
            thread_id="th_e",
            server_id="srv_e",
            tool_name="tool_e",
            arguments={"key": "val"},
            category="READ",
            reason="Test E User",
        )
        mgr.decide(rec.approval_id, user_id="alice", approved=True)

        # Bob attempts to consume Alice's approval
        assert mgr.verify_and_consume(
            rec.approval_id, "bob", "th_e", "srv_e", "tool_e", {"key": "val"}
        ) is False

        # Alice's record remains APPROVED
        rec_after = mgr.repo.get(rec.approval_id, "alice")
        assert rec_after is not None
        assert rec_after.status == "APPROVED"
    finally:
        db.close()


def test_atomic_approval_consumption_mismatched_thread_server_tool(tmp_path: Path) -> None:
    """Test F: Wrong thread / server / tool rejected: returns False, record remains APPROVED."""
    db = DatabaseManager(db_path=tmp_path / "test_f.db")
    try:
        mgr = ApprovalManager(db_manager=db)
        rec = mgr.create_pending_approval(
            user_id="user_f",
            thread_id="th_f",
            server_id="srv_f",
            tool_name="tool_f",
            arguments={"query": "status"},
            category="READ",
            reason="Test F Context",
        )
        mgr.decide(rec.approval_id, user_id="user_f", approved=True)

        # Mismatched thread_id
        assert mgr.verify_and_consume(
            rec.approval_id, "user_f", "th_other", "srv_f", "tool_f", {"query": "status"}
        ) is False

        # Mismatched server_id
        assert mgr.verify_and_consume(
            rec.approval_id, "user_f", "th_f", "srv_other", "tool_f", {"query": "status"}
        ) is False

        # Mismatched tool_name
        assert mgr.verify_and_consume(
            rec.approval_id, "user_f", "th_f", "srv_f", "tool_other", {"query": "status"}
        ) is False

        # Record remains APPROVED
        rec_after = mgr.repo.get(rec.approval_id, "user_f")
        assert rec_after is not None
        assert rec_after.status == "APPROVED"

        # Correct call succeeds
        assert mgr.verify_and_consume(
            rec.approval_id, "user_f", "th_f", "srv_f", "tool_f", {"query": "status"}
        ) is True
    finally:
        db.close()


def test_atomic_approval_consumption_invalid_states(tmp_path: Path) -> None:
    """Test G: PENDING, REJECTED, and CONSUMED approvals cannot be consumed."""
    db = DatabaseManager(db_path=tmp_path / "test_g.db")
    try:
        mgr = ApprovalManager(db_manager=db)

        # 1. PENDING state
        pending_rec = mgr.create_pending_approval(
            user_id="user_g",
            thread_id="th_g",
            server_id="srv_g",
            tool_name="tool_g",
            arguments={"step": 1},
            category="READ",
            reason="Test G Pending",
        )
        assert mgr.verify_and_consume(
            pending_rec.approval_id, "user_g", "th_g", "srv_g", "tool_g", {"step": 1}
        ) is False

        # 2. REJECTED state
        rejected_rec = mgr.create_pending_approval(
            user_id="user_g",
            thread_id="th_g",
            server_id="srv_g",
            tool_name="tool_g",
            arguments={"step": 2},
            category="READ",
            reason="Test G Rejected",
        )
        mgr.decide(rejected_rec.approval_id, user_id="user_g", approved=False)
        assert mgr.verify_and_consume(
            rejected_rec.approval_id, "user_g", "th_g", "srv_g", "tool_g", {"step": 2}
        ) is False

        # 3. CONSUMED state
        consumed_rec = mgr.create_pending_approval(
            user_id="user_g",
            thread_id="th_g",
            server_id="srv_g",
            tool_name="tool_g",
            arguments={"step": 3},
            category="READ",
            reason="Test G Consumed",
        )
        mgr.decide(consumed_rec.approval_id, user_id="user_g", approved=True)
        # First consume -> True
        assert mgr.verify_and_consume(
            consumed_rec.approval_id, "user_g", "th_g", "srv_g", "tool_g", {"step": 3}
        ) is True
        # Already CONSUMED -> False
        assert mgr.verify_and_consume(
            consumed_rec.approval_id, "user_g", "th_g", "srv_g", "tool_g", {"step": 3}
        ) is False
    finally:
        db.close()


def test_atomic_approval_consumption_db_failure(tmp_path: Path) -> None:
    """Test H: Database failure handling: verify_and_consume returns False safely without partial state corruption."""
    db = DatabaseManager(db_path=tmp_path / "test_h.db")
    try:
        mgr = ApprovalManager(db_manager=db)
        rec = mgr.create_pending_approval(
            user_id="user_h",
            thread_id="th_h",
            server_id="srv_h",
            tool_name="tool_h",
            arguments={"action": "test_err"},
            category="READ",
            reason="Test H Error",
        )
        mgr.decide(rec.approval_id, user_id="user_h", approved=True)

        # Simulate database failure during execute
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = sqlite3.OperationalError("database disk image is malformed")
        with patch.object(DatabaseManager, "connection", new_callable=PropertyMock, return_value=mock_conn):
            res = mgr.verify_and_consume(
                rec.approval_id, "user_h", "th_h", "srv_h", "tool_h", {"action": "test_err"}
            )
            assert res is False

        # After error, DB record is intact and still APPROVED
        rec_after = mgr.repo.get(rec.approval_id, "user_h")
        assert rec_after is not None
        assert rec_after.status == "APPROVED"
    finally:
        db.close()

