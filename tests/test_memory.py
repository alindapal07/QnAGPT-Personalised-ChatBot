"""Pytest suite for SQLite checkpointing, multi-thread persistence, and user isolation."""

from pathlib import Path

import pytest

from qna_bot.agent.memory import (
    ConversationMemoryManager,
    create_new_thread,
    retrieve_all_threads,
    retrieve_chat_history,
)
from qna_bot.core.exceptions import MemoryPersistenceError


def test_thread_lifecycle_persistence_and_isolation(tmp_path: Path) -> None:
    db_file = tmp_path / "chatbot.db"
    mem = ConversationMemoryManager(db_path=db_file)

    t1 = create_new_thread(user_id="alice", memory=mem, title="First")
    t2 = create_new_thread(user_id="alice", memory=mem, title="Second")
    t_bob = create_new_thread(user_id="bob", memory=mem, title="Bob Thread")

    assert t1 != t2

    mem.append_message(
        t1,
        user_id="alice",
        message={
            "role": "user",
            "content": "What is Agentic RAG?",
            "sources": [],
            "web_sources": [],
        },
    )
    mem.append_message(
        t1,
        user_id="alice",
        message={
            "role": "assistant",
            "content": "Agentic RAG combines routing and retrieval.",
            "sources": [{"source": "rag.pdf", "page": 2}],
            "web_sources": [],
        },
    )
    mem.update_thread_title(t1, user_id="alice", first_prompt="What is Agentic RAG?")
    mem.append_message(
        t2,
        user_id="alice",
        message={"role": "user", "content": "Independent thread 2"},
    )
    mem.append_message(
        t_bob,
        user_id="bob",
        message={"role": "user", "content": "Bob private chat"},
    )
    mem.close()

    # Reopen from disk to prove persistence
    reopened = ConversationMemoryManager(db_path=db_file)
    try:
        alice_threads = retrieve_all_threads(user_id="alice", memory=reopened)
        assert len(alice_threads) == 2
        assert all(t["thread_id"] != t_bob for t in alice_threads)

        history = retrieve_chat_history(t1, user_id="alice", memory=reopened)
        assert len(history) == 2
        assert history[1]["sources"] == [{"source": "rag.pdf", "page": 2}]

        # Bob cannot read or delete Alice's thread
        with pytest.raises(MemoryPersistenceError):
            retrieve_chat_history(t1, user_id="bob", memory=reopened)
        with pytest.raises(MemoryPersistenceError):
            reopened.delete_thread(t1, user_id="bob")

        # Invalid thread IDs fail cleanly
        with pytest.raises(MemoryPersistenceError):
            retrieve_chat_history("", user_id="alice", memory=reopened)

        assert reopened.delete_thread(t2, user_id="alice") is True
        assert len(retrieve_all_threads(user_id="alice", memory=reopened)) == 1
    finally:
        reopened.close()
