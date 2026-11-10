"""Presentation layer exports."""

from qna_bot.ui.agent_workspace import render_agent_workspace_page
from qna_bot.ui.auth_views import (
    handle_oauth_callback_params,
    render_account_page,
    render_auth_gate,
)
from qna_bot.ui.chat_view import (
    render_ai_message,
    render_chat_history,
    render_user_message,
)
from qna_bot.ui.components import (
    render_approval_card,
    render_copy_button,
    render_header,
    render_sources_panel,
    render_welcome_screen,
)
from qna_bot.ui.composer import ComposerSubmission, render_chat_composer
from qna_bot.ui.design_system import (
    render_empty_state,
    render_execution_timeline_html,
    render_metric_strip,
    render_page_header,
    render_section_header,
    status_pill_html,
)
from qna_bot.ui.mcp_dashboard import render_mcp_dashboard
from qna_bot.ui.mcp_marketplace import render_mcp_marketplace_page
from qna_bot.ui.sidebar import WORKSPACE_PAGES, SidebarConfig, render_sidebar
from qna_bot.ui.styles import apply_styles
from qna_bot.ui.workspace_views import (
    render_knowledge_base_page,
    render_settings_page,
    render_tool_marketplace_page,
)

__all__ = [
    "ComposerSubmission",
    "SidebarConfig",
    "WORKSPACE_PAGES",
    "apply_styles",
    "handle_oauth_callback_params",
    "render_account_page",
    "render_agent_workspace_page",
    "render_ai_message",
    "render_approval_card",
    "render_auth_gate",
    "render_chat_composer",
    "render_chat_history",
    "render_copy_button",
    "render_empty_state",
    "render_execution_timeline_html",
    "render_header",
    "render_knowledge_base_page",
    "render_mcp_dashboard",
    "render_mcp_marketplace_page",
    "render_metric_strip",
    "render_page_header",
    "render_section_header",
    "render_settings_page",
    "render_sidebar",
    "render_sources_panel",
    "render_tool_marketplace_page",
    "render_user_message",
    "render_welcome_screen",
    "status_pill_html",
]
