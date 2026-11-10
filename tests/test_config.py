"""Tests for application configuration and environment settings."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from qna_bot.config.settings import AppSettings, get_settings, load_settings


class TestConfigSettings(unittest.TestCase):
    def test_defaults(self) -> None:
        settings = AppSettings(groq_api_key="test-key")
        self.assertEqual(settings.default_text_model, "openai/gpt-oss-20b")
        self.assertEqual(settings.default_vision_model, "qwen/qwen3.8-27b")
        self.assertIn("openai/gpt-oss-120b", settings.available_text_models)
        self.assertEqual(settings.default_temperature, 0.7)
        self.assertEqual(settings.default_max_tokens, 2048)
        self.assertEqual(settings.max_image_size_mb, 10)
        self.assertEqual(settings.allowed_image_types, ("png", "jpg", "jpeg", "webp"))

    def test_load_settings_reads_env(self) -> None:
        with patch.dict(
            os.environ,
            {"GROQ_API_KEY": "  gsk_demo_123  ", "LOG_LEVEL": "debug"},
            clear=False,
        ):
            settings = load_settings()
        self.assertEqual(settings.groq_api_key, "gsk_demo_123")
        self.assertEqual(settings.log_level, "debug")

    def test_get_settings_returns_cached_instance(self) -> None:
        get_settings.cache_clear()
        first = get_settings()
        second = get_settings()
        self.assertIs(first, second)
        get_settings.cache_clear()

    def test_dotenv_loads_glama_key_even_when_groq_key_is_preset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("GLAMA_API_KEY=glama_unit_test_val\n", encoding="utf-8")
            with (
                patch("qna_bot.config.settings._candidate_env_paths", return_value=(env_file,)),
                patch.dict(os.environ, {"GROQ_API_KEY": "gsk_preset", "GLAMA_API_KEY": ""}, clear=False),
            ):
                loaded = load_settings()
                self.assertEqual(loaded.groq_api_key, "gsk_preset")
                self.assertEqual(loaded.glama_api_key, "glama_unit_test_val")


if __name__ == "__main__":
    unittest.main()
