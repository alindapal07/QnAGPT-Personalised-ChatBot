"""Tests for LangChain message construction and AIAgent routing/streaming."""

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from qna_bot.core.exceptions import AgentExecutionError, ConfigurationError
from qna_bot.services.agent_service import (
    AIAgent,
    build_langchain_messages,
    stream_chunks,
)


class TestAgentService(unittest.TestCase):
    def test_build_langchain_messages_text_and_multimodal(self) -> None:
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {
                "role": "user",
                "content": "",
                "image_b64": "ZmFrZQ==",
                "image_mime": "image/jpeg",
            },
        ]
        lc_msgs = build_langchain_messages(history, "Be helpful.")
        self.assertEqual(len(lc_msgs), 4)
        self.assertIsInstance(lc_msgs[0], SystemMessage)
        self.assertEqual(lc_msgs[0].content, "Be helpful.")
        self.assertIsInstance(lc_msgs[1], HumanMessage)
        self.assertEqual(lc_msgs[1].content, "Hello")
        self.assertIsInstance(lc_msgs[2], AIMessage)
        self.assertIsInstance(lc_msgs[3], HumanMessage)
        self.assertEqual(lc_msgs[3].content[0]["text"], "Please analyze this image.")
        self.assertEqual(
            lc_msgs[3].content[1]["image_url"]["url"],
            "data:image/jpeg;base64,ZmFrZQ==",
        )

    def test_build_langchain_messages_skips_blank_system_prompt(self) -> None:
        lc_msgs = build_langchain_messages([{"role": "user", "content": "Ping"}], "   ")
        self.assertEqual(len(lc_msgs), 1)
        self.assertIsInstance(lc_msgs[0], HumanMessage)

    def test_stream_chunks_supports_objects_strings_and_skips_empty(self) -> None:
        chunks = [
            SimpleNamespace(content="Hello "),
            SimpleNamespace(content=""),
            "world",
            SimpleNamespace(content=[{"type": "text", "text": "!"}]),
        ]
        out = list(stream_chunks(chunks, delay_sec=0))
        self.assertEqual(out, ["Hello ", "world", "!"])

    def test_active_model_routing(self) -> None:
        agent = AIAgent(
            text_model="openai/gpt-oss-20b",
            vision_model="qwen/qwen3.8-27b",
            api_key="test-key",
            stream_delay_sec=0,
        )
        self.assertEqual(agent.active_model_name(has_image=False), "openai/gpt-oss-20b")
        self.assertEqual(agent.active_model_name(has_image=True), "qwen/qwen3.8-27b")

    def test_stream_missing_api_key_raises_configuration_error(self) -> None:
        agent = AIAgent(
            text_model="openai/gpt-oss-20b",
            api_key="",
            stream_delay_sec=0,
        )
        agent.api_key = ""
        with self.assertRaises(ConfigurationError):
            list(agent.stream([{"role": "user", "content": "Hi"}]))

    def test_stream_invokes_routed_model_and_yields_tokens(self) -> None:
        mock_llm = MagicMock()
        mock_llm.stream.return_value = [
            SimpleNamespace(content="Chunk 1 "),
            SimpleNamespace(content="Chunk 2"),
        ]
        agent = AIAgent(
            text_model="openai/gpt-oss-20b",
            vision_model="qwen/qwen3.8-27b",
            api_key="gsk_valid",
            stream_delay_sec=0,
        )

        with patch.object(agent, "_make_llm", return_value=mock_llm) as mock_factory:
            tokens = list(
                agent.stream(
                    [{"role": "user", "content": "Describe", "image_b64": "abc"}],
                    has_image=True,
                )
            )

        mock_factory.assert_called_once_with("qwen/qwen3.8-27b")
        self.assertEqual(tokens, ["Chunk 1 ", "Chunk 2"])

    def test_stream_wraps_provider_errors(self) -> None:
        mock_llm = MagicMock()
        mock_llm.stream.side_effect = ValueError("API error")
        agent = AIAgent(
            text_model="openai/gpt-oss-20b",
            api_key="gsk_valid",
            stream_delay_sec=0,
        )

        with patch.object(agent, "_make_llm", return_value=mock_llm):
            with self.assertRaises(AgentExecutionError):
                list(agent.stream([{"role": "user", "content": "Hello"}]))


if __name__ == "__main__":
    unittest.main()
