"""Tests for conversation export serialization."""

import json
import unittest

from qna_bot.services.export_service import (
    export_as_json,
    export_as_markdown,
    export_as_text,
)


class TestExportService(unittest.TestCase):
    def setUp(self) -> None:
        self.messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

    def test_export_as_json(self) -> None:
        output = export_as_json("chat-42", self.messages)
        parsed = json.loads(output)
        self.assertEqual(parsed["chat_id"], "chat-42")
        self.assertEqual(len(parsed["messages"]), 2)
        self.assertEqual(parsed["messages"][1]["content"], "Hi there!")

    def test_export_as_markdown(self) -> None:
        output = export_as_markdown(self.messages)
        self.assertIn("# Chat Export", output)
        self.assertIn("### User\nHello", output)
        self.assertIn("### Assistant\nHi there!", output)

    def test_export_as_text(self) -> None:
        output = export_as_text(self.messages)
        self.assertEqual(output, "USER: Hello\nASSISTANT: Hi there!")


if __name__ == "__main__":
    unittest.main()
