"""Tests for UI assets and application orchestration flow."""

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from qna_bot import app as qna_app
from qna_bot.config.settings import AppSettings
from qna_bot.services.session_service import initialize_session_state
from qna_bot.ui.sidebar import SidebarConfig
from qna_bot.ui.styles import load_theme_markup


class _AttrDict(dict):
    """Dict supporting attribute access like Streamlit's st.session_state."""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value) -> None:
        self[name] = value


class _DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class TestUIAssetsAndAppFlow(unittest.TestCase):
    def test_theme_assets_contain_expected_tokens(self) -> None:
        load_theme_markup.cache_clear()
        markup = load_theme_markup()
        self.assertIn("--app-background", markup)
        self.assertIn('[data-theme="dark"]', markup)
        self.assertIn(".dot-ok", markup)
        self.assertIn(".dot-err", markup)
        self.assertIn("detectAndApplyTheme", markup)
        self.assertIn(r"/\d+/g", markup)

    def test_create_agent_maps_sidebar_and_settings(self) -> None:
        settings = AppSettings(groq_api_key="gsk_test", stream_chunk_delay_sec=0.0)
        sidebar_cfg = SidebarConfig(
            text_model="openai/gpt-oss-120b",
            vision_model="qwen/qwen3.8-27b",
            temperature=0.3,
            max_tokens=1024,
            system_prompt="Custom prompt",
        )
        agent = qna_app.create_agent(sidebar_cfg, settings)
        self.assertEqual(agent.text_model, "openai/gpt-oss-120b")
        self.assertEqual(agent.vision_model, "qwen/qwen3.8-27b")
        self.assertEqual(agent.temperature, 0.3)
        self.assertEqual(agent.max_tokens, 1024)
        self.assertEqual(agent.api_key, "gsk_test")
        self.assertEqual(agent.system_prompt, "Custom prompt")
        self.assertEqual(agent.stream_delay_sec, 0.0)

    def test_submit_user_message_appends_user_and_assistant_messages(self) -> None:
        fake_state = _AttrDict()
        initialize_session_state(fake_state)
        fake_state["pending_image"] = {
            "name": "chart.png",
            "b64": "ZmFrZQ==",
            "mime": "image/png",
            "format": "png",
            "width": 10,
            "height": 10,
        }

        mock_agent = MagicMock()
        mock_agent.stream.return_value = iter(["Analyzed ", "chart."])

        fake_st = SimpleNamespace(
            session_state=fake_state,
            columns=lambda spec: (_DummyContext(), _DummyContext()),
            markdown=MagicMock(),
            chat_message=lambda role: _DummyContext(),
            spinner=lambda label: _DummyContext(),
            write_stream=lambda gen: "".join(gen),
            error=MagicMock(),
            expander=lambda label: _DummyContext(),
            code=MagicMock(),
            rerun=MagicMock(),
        )

        with (
            patch.object(qna_app, "st", fake_st),
            patch.object(qna_app, "render_user_message") as mock_render_user,
        ):
            qna_app.submit_user_message(mock_agent, "Explain this chart")

        self.assertIsNone(fake_state["pending_image"])
        self.assertEqual(fake_state["uploader_nonce"], 1)
        self.assertEqual(len(fake_state["messages"]), 2)
        self.assertEqual(fake_state["messages"][0]["role"], "user")
        self.assertEqual(fake_state["messages"][0]["content"], "Explain this chart")
        self.assertEqual(fake_state["messages"][0]["image_b64"], "ZmFrZQ==")
        self.assertEqual(fake_state["messages"][1]["role"], "assistant")
        self.assertEqual(fake_state["messages"][1]["content"], "Analyzed chart.")
        mock_agent.stream.assert_called_once()
        self.assertTrue(mock_agent.stream.call_args.kwargs["has_image"])
        mock_render_user.assert_called_once()
        fake_st.rerun.assert_called_once()

    def test_process_pending_prompt_handles_suggestion_and_regenerate(self) -> None:
        fake_state = _AttrDict()
        initialize_session_state(fake_state)
        fake_state["prompt_to_send"] = "Explain LangChain"

        fake_st = SimpleNamespace(session_state=fake_state, rerun=MagicMock())
        with patch.object(qna_app, "st", fake_st):
            prompt = qna_app.process_pending_prompt(MagicMock())

        self.assertEqual(prompt, "Explain LangChain")
        self.assertIsNone(fake_state["prompt_to_send"])
        fake_st.rerun.assert_not_called()

        # Now test regeneration path
        fake_state["messages"] = [
            {"role": "user", "content": "See image", "image_b64": "YWJj"}
        ]
        fake_state["prompt_to_send"] = "__REGENERATE__"
        mock_agent = MagicMock()

        with (
            patch.object(qna_app, "st", fake_st),
            patch.object(qna_app, "execute_ai_turn") as mock_exec,
        ):
            qna_app.process_pending_prompt(mock_agent)

        mock_exec.assert_called_once_with(mock_agent, has_image=True)
        fake_st.rerun.assert_called_once()


if __name__ == "__main__":
    unittest.main()
