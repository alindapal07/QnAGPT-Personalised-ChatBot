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
        medium_prompt = "A" * 50
        state["messages"].append({"role": "user", "content": medium_prompt})
        save_current_chat_to_recent(state)

        self.assertEqual(len(state["recent_chats"]), 1)
        self.assertEqual(state["recent_chats"][0]["title"], medium_prompt)

        state["messages"].append({"role": "assistant", "content": "Response"})
        save_current_chat_to_recent(state)
        self.assertEqual(len(state["recent_chats"]), 1)
        self.assertEqual(len(state["recent_chats"][0]["messages"]), 2)

        long_state = {}
        initialize_session_state(long_state)
        long_state["messages"].append({"role": "user", "content": "B" * 150})
        save_current_chat_to_recent(long_state)
        self.assertEqual(long_state["recent_chats"][0]["title"], ("B" * 120) + "...")

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

    def test_derive_chat_title_multimodal_list_content(self) -> None:
        from qna_bot.services.session_service import _derive_chat_title

        # Multimodal message with text and image
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the financial charts in this image"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
                ],
            }
        ]
        title = _derive_chat_title(msgs)
        self.assertIsInstance(title, str)
        self.assertEqual(title, "Analyze the financial charts in this image")

        # Multimodal message with image only (no text part)
        img_only = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
                ],
            }
        ]
        self.assertEqual(_derive_chat_title(img_only), "Image Analysis")

        # Multimodal message with empty content list
        empty_list = [{"role": "user", "content": []}]
        self.assertEqual(_derive_chat_title(empty_list), "Conversation")

        # Unknown block types with no text and no image
        unknown_blocks = [{"role": "user", "content": [{"type": "custom", "payload": 123}]}]
        self.assertEqual(_derive_chat_title(unknown_blocks), "Conversation")

    def test_derive_chat_title_empty_and_whitespace_fallback(self) -> None:
        from qna_bot.services.session_service import _derive_chat_title

        self.assertEqual(_derive_chat_title([]), "Conversation")
        self.assertEqual(_derive_chat_title([{"role": "user", "content": "   \n\t  "}]), "Conversation")
        self.assertEqual(_derive_chat_title([{"role": "assistant", "content": "Hi"}]), "Conversation")
        self.assertEqual(_derive_chat_title([{"role": "user", "content": "Multi\nLine\tTitle"}]), "Multi Line Title")



if __name__ == "__main__":
    unittest.main()
