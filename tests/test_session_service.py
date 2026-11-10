"""Tests for session state initialization, history saving, and chat transitions."""

import unittest

from qna_bot.services.session_service import (
    clear_current_chat,
    initialize_session_state,
    restore_chat,
    save_current_chat_to_recent,
    start_new_chat,
)


class TestSessionService(unittest.TestCase):
    def test_initialize_session_state_sets_defaults_without_overwriting(self) -> None:
        state = {"search_query": "langchain"}
        initialize_session_state(state)
        self.assertEqual(state["search_query"], "langchain")
        self.assertEqual(state["messages"], [])
        self.assertEqual(state["recent_chats"], [])
        self.assertIn("current_chat_id", state)
        self.assertEqual(state["uploader_nonce"], 0)

    def test_save_and_update_recent_chat(self) -> None:
        state = {}
        initialize_session_state(state)
        long_prompt = "A" * 50
        state["messages"].append({"role": "user", "content": long_prompt})
        save_current_chat_to_recent(state)

        self.assertEqual(len(state["recent_chats"]), 1)
        self.assertEqual(state["recent_chats"][0]["title"], ("A" * 36) + "...")

        state["messages"].append({"role": "assistant", "content": "Response"})
        save_current_chat_to_recent(state)
        self.assertEqual(len(state["recent_chats"]), 1)
        self.assertEqual(len(state["recent_chats"][0]["messages"]), 2)

    def test_start_new_chat_archives_previous_and_resets(self) -> None:
        state = {}
        initialize_session_state(state)
        old_id = state["current_chat_id"]
        state["messages"] = [{"role": "user", "content": "First chat"}]
        state["feedback"] = {0: "helpful"}
        state["pending_image"] = {"name": "img.png"}

        start_new_chat(state)

        self.assertNotEqual(state["current_chat_id"], old_id)
        self.assertEqual(state["messages"], [])
        self.assertIsNone(state["pending_image"])
        self.assertEqual(state["feedback"], {})
        self.assertEqual(state["uploader_nonce"], 1)
        self.assertEqual(len(state["recent_chats"]), 1)

    def test_clear_current_chat_does_not_add_recent(self) -> None:
        state = {}
        initialize_session_state(state)
        state["messages"] = [{"role": "user", "content": "Temp"}]
        clear_current_chat(state)
        self.assertEqual(state["messages"], [])
        self.assertEqual(state["recent_chats"], [])
        self.assertEqual(state["uploader_nonce"], 1)

    def test_restore_chat(self) -> None:
        state = {}
        initialize_session_state(state)
        chat = {
            "id": "restored-1",
            "title": "Old",
            "messages": [{"role": "user", "content": "Past question"}],
        }
        restore_chat(chat, state)
        self.assertEqual(state["current_chat_id"], "restored-1")
        self.assertEqual(state["messages"], [{"role": "user", "content": "Past question"}])


if __name__ == "__main__":
    unittest.main()
