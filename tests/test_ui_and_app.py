"""Tests for Light-mode UI assets, conversation grouping, and application orchestration flow."""

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import time
import unittest
from unittest.mock import MagicMock, patch

from qna_bot import app as qna_app
from qna_bot.agent.memory import ConversationMemoryManager, classify_time_bucket
from qna_bot.config.settings import AppSettings
from qna_bot.services.conversation_service import ConversationService
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
    def test_theme_assets_are_light_mode_only(self) -> None:
        load_theme_markup.cache_clear()
        markup = load_theme_markup()
        self.assertIn("--app-background:      #FFFFFF;", markup)
        self.assertIn(".dot-ok", markup)
        self.assertIn(".dot-err", markup)
        self.assertIn(".approval-card", markup)
        # Ensure dark mode CSS and JS theme switcher were completely removed
        self.assertNotIn('[data-theme="dark"]', markup)
        self.assertNotIn("prefers-color-scheme: dark", markup)
        self.assertNotIn("detectAndApplyTheme", markup)

    def test_conversation_time_buckets_rename_pin_and_content_search(self) -> None:
        now = 1_700_000_000.0
        self.assertEqual(classify_time_bucket(now, now_ts=now), "Today")
        self.assertEqual(classify_time_bucket(now - 86400, now_ts=now), "Yesterday")
        self.assertEqual(classify_time_bucket(now - 4 * 86400, now_ts=now), "Previous 7 Days")
        self.assertEqual(classify_time_bucket(now - 20 * 86400, now_ts=now), "Previous 30 Days")
        self.assertEqual(classify_time_bucket(now - 45 * 86400, now_ts=now), "Older")

        mem = ConversationMemoryManager(db_path=Path(":memory:"))
        try:
            svc = ConversationService(mem)
            t1 = svc.create_thread("alice", title="Alpha Thread")
            mem.append_message(t1, "alice", {"role": "user", "content": "Discuss FAISS vector storage"})
            svc.rename_thread(t1, "alice", "Renamed Vector Chat")
            svc.set_pinned(t1, "alice", True)

            grouped = svc.list_threads_grouped("alice", search_query="FAISS")
            self.assertIn("Pinned", grouped)
            self.assertEqual(grouped["Pinned"][0]["title"], "Renamed Vector Chat")
            self.assertEqual(svc.list_threads_grouped("alice", search_query="nonexistent"), {})
        finally:
            mem.close()

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
            caption=MagicMock(),
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
        fake_state["prompt_to_send"] = "Analyze my documents"

        fake_st = SimpleNamespace(session_state=fake_state, rerun=MagicMock())
        with patch.object(qna_app, "st", fake_st):
            prompt = qna_app.process_pending_prompt(MagicMock())

        self.assertEqual(prompt, "Analyze my documents")
        self.assertIsNone(fake_state["prompt_to_send"])
        fake_st.rerun.assert_not_called()

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

        mock_exec.assert_called_once_with(
            mock_agent,
            has_image=True,
            chat_service=None,
            sidebar_cfg=None,
        )
        fake_st.rerun.assert_called_once()

    def test_shadowed_sys_path_does_not_cause_mcp_circular_import(self) -> None:
        pkg_dir = Path(__file__).resolve().parent.parent / "qna_bot"
        script = (
            "import sys; "
            f"sys.path.insert(0, {str(pkg_dir)!r}); "
            "from qna_bot.agent.approval import ApprovalManager; "
            "from qna_bot.mcp_client.client_manager import MCPClientManager; "
            "import mcp; "
            "print(mcp.__file__); "
            f"assert 'site-packages' in mcp.__file__ or {str(pkg_dir / 'mcp')!r} not in mcp.__file__"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(pkg_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}")

    def test_lazy_package_exports_resolve_cleanly(self) -> None:
        import qna_bot
        import qna_bot.agent
        import qna_bot.mcp_client
        import qna_bot.services

        for pkg in (qna_bot, qna_bot.mcp_client, qna_bot.agent, qna_bot.services):
            for name in pkg.__all__:
                self.assertTrue(hasattr(pkg, name), f"Missing export {name} on {pkg.__name__}")

    def test_design_system_timeline_and_svg_navigation(self) -> None:
        from qna_bot.ui.design_system import COLORS, render_execution_timeline_html, status_pill_html
        from qna_bot.ui.sidebar import NAV_GROUPS, WORKSPACE_PAGES

        self.assertEqual(COLORS["bg_main"], "#FFFFFF")
        self.assertEqual(COLORS["accent"], "#315EAB")
        self.assertEqual(COLORS["text_primary"], "#172033")

        flat_pages = [p for _, items in NAV_GROUPS for p, _ in items]
        self.assertEqual(tuple(flat_pages), WORKSPACE_PAGES)
        for _, items in NAV_GROUPS:
            for page_name, icon_name in items:
                self.assertTrue(icon_name.startswith(":material/"), f"Expected Material SVG icon for {page_name}")

        tl_html = render_execution_timeline_html(
            [
                {"label": "Web search", "detail": "4 results found", "state": "completed"},
                {"label": "Retrieving webpage content", "detail": "In progress", "state": "in_progress"},
                {"label": "Generating response", "detail": "", "state": "pending"},
            ]
        )
        self.assertIn("<svg", tl_html)
        self.assertIn("4 results found", tl_html)
        self.assertIn("tl-label-active", tl_html)
        self.assertIn("status-pill-ok", status_pill_html("Connected", "ok"))

        # Verify collapsible sidebar CSS does not lock collapsed width or omit mobile drawer rules
        css_markup = load_theme_markup()
        self.assertIn('[aria-expanded="false"]', css_markup)
        self.assertIn("max-width: 768px", css_markup)
        self.assertIn("text-overflow: ellipsis", css_markup)

        # Ensure Settings, Knowledge Base, Auth, Agent Workspace, and Dashboard use render_html instead of raw indented divs
        ui_dir = Path(__file__).resolve().parent.parent / "qna_bot" / "ui"
        for fname in (
            "workspace_views.py",
            "agent_workspace.py",
            "auth_views.py",
            "mcp_dashboard.py",
            "mcp_marketplace.py",
            "sidebar.py",
            "components.py",
        ):
            src = (ui_dir / fname).read_text(encoding="utf-8")
            self.assertNotIn("unsafe_allow_html", src, msg=f"Unexpected raw unsafe_allow_html in {fname}")

    def test_app_renders_all_workspace_pages_cleanly(self) -> None:
        import os
        import tempfile
        from streamlit.testing.v1 import AppTest
        from qna_bot.config.settings import get_settings

        app_path = str(Path(__file__).resolve().parent.parent / "qna_bot" / "app.py")
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            prev_data_dir = os.environ.get("QNAGPT_DATA_DIR")
            prev_qdrant_mode = os.environ.get("QDRANT_MODE")
            prev_qdrant_url = os.environ.get("QDRANT_URL")
            prev_qdrant_key = os.environ.get("QDRANT_API_KEY")
            os.environ["QNAGPT_DATA_DIR"] = tmp_dir
            os.environ["QDRANT_MODE"] = "local"
            os.environ["QDRANT_URL"] = "http://localhost:6333"
            os.environ["QDRANT_API_KEY"] = ""
            get_settings.cache_clear()
            try:
                at = AppTest.from_file(app_path, default_timeout=20)
                at.run()
                self.assertFalse(at.exception, msg=f"Auth gate exception: {at.exception}")
                # Unauthenticated users cannot see protected workspace navigation buttons
                self.assertEqual([b for b in at.sidebar.button if str(b.key).startswith("nav_page_")], [])

                # Register a new authenticated user through the real Auth UI form
                at.text_input(key="auth_reg_name").set_value("UI Tester")
                at.text_input(key="auth_reg_email").set_value("tester@qnagpt.local")
                at.text_input(key="auth_reg_pw").set_value("TestPass123!")
                at.text_input(key="auth_reg_pw2").set_value("TestPass123!")
                reg_submit = next(b for b in at.button if b.label == "Create Account")
                reg_submit.click().run()
                self.assertFalse(at.exception, msg=f"Chat page exception after sign-up: {at.exception}")

                # Verify all workspace pages under deterministic mock without live registry network requests
                with patch(
                    "qna_bot.mcp_client.registry.service.MCPRegistryService.discover",
                    return_value=SimpleNamespace(
                        items=[],
                        attributions=[("Official MCP Registry", "https://registry.modelcontextprotocol.io")],
                        warning="",
                        has_next_page=False,
                        next_cursor=None,
                    ),
                ):
                    for target_page in (
                        "Agent Workspace",
                        "Knowledge Base",
                        "MCP Servers",
                        "Tool Marketplace",
                        "Account",
                        "Settings",
                        "MCP Marketplace",
                    ):
                        nav_btn = next(b for b in at.sidebar.button if b.key == f"nav_page_{target_page}")
                        nav_btn.click().run()
                        self.assertFalse(at.exception, msg=f"{target_page} exception: {at.exception}")

                    # Navigate BACK to Chat to verify header and composer rendering
                    chat_btn = next(b for b in at.sidebar.button if b.key == "nav_page_Chat")
                    chat_btn.click().run()
                    self.assertFalse(at.exception, msg=f"Return to Chat exception: {at.exception}")

                    # Sign out from Account page to verify return to Auth gate
                    acct_btn = next(b for b in at.sidebar.button if b.key == "nav_page_Account")
                    acct_btn.click().run()
                    self.assertFalse(at.exception, msg=f"Account nav exception before signout: {at.exception}")
                    sign_out_btn = next((b for b in at.button if "Sign Out" in str(b.label)), None)
                    if sign_out_btn is not None:
                        sign_out_btn.click().run()
                        self.assertFalse(at.exception, msg=f"Sign-out exception: {at.exception}")
                        # Auth gate should be displayed again after sign out
                        self.assertEqual([b for b in at.sidebar.button if str(b.key).startswith("nav_page_")], [])
            finally:
                try:
                    import streamlit as st

                    db_path = Path(tmp_dir) / "chatbot.db"
                    qna_app._get_shared_mcp_manager(str(db_path), 15, 60).shutdown()
                    qna_app._get_shared_memory(str(db_path)).close()
                    qna_app._get_shared_orm_db(f"sqlite:///{db_path.resolve().as_posix()}").close()
                    st.cache_resource.clear()
                except Exception:
                    pass
                if prev_data_dir is None:
                    os.environ.pop("QNAGPT_DATA_DIR", None)
                else:
                    os.environ["QNAGPT_DATA_DIR"] = prev_data_dir

                if prev_qdrant_mode is None:
                    os.environ.pop("QDRANT_MODE", None)
                else:
                    os.environ["QDRANT_MODE"] = prev_qdrant_mode

                if prev_qdrant_url is None:
                    os.environ.pop("QDRANT_URL", None)
                else:
                    os.environ["QDRANT_URL"] = prev_qdrant_url

                if prev_qdrant_key is None:
                    os.environ.pop("QDRANT_API_KEY", None)
                else:
                    os.environ["QDRANT_API_KEY"] = prev_qdrant_key
                get_settings.cache_clear()

    def test_navigation_fallback_for_invalid_active_page(self) -> None:
        import streamlit as st
        from qna_bot.ui.sidebar import _render_navigation, WORKSPACE_PAGES

        # Mock st.session_state with an unrecognized page name
        with patch.object(st, "session_state", {"active_page": "NonExistentPage"}):
            with patch.object(st, "button", return_value=False):
                resolved_page = _render_navigation()
                self.assertEqual(resolved_page, "Chat")

        # Test with empty string
        with patch.object(st, "session_state", {"active_page": ""}):
            with patch.object(st, "button", return_value=False):
                resolved_page = _render_navigation()
                self.assertEqual(resolved_page, "Chat")

        # Test with None
        with patch.object(st, "session_state", {}):
            with patch.object(st, "button", return_value=False):
                resolved_page = _render_navigation()
                self.assertEqual(resolved_page, "Chat")

    def test_render_header_accepts_all_workspace_titles_cleanly(self) -> None:
        from qna_bot.ui.components import render_header
        from qna_bot.config.settings import get_settings

        settings = get_settings()
        # Verify render_header executes without error for Chat and other page titles
        for page_name in ("Chat", "Agent Workspace", "Knowledge Base", "MCP Servers", "Settings"):
            render_header(
                model_name="llama-3.3-70b-versatile",
                chat_mode="Auto (Agentic)",
                page_title=page_name,
                conversation_title="Active Conversation",
                settings=settings,
            )



if __name__ == "__main__":
    unittest.main()
