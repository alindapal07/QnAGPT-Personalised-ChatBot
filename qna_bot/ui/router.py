"""Workspace page router dispatching between top-level views."""

from typing import Any, Callable, Dict, Optional

from qna_bot.config.settings import AppSettings
from qna_bot.services.approval_service import ApprovalService
from qna_bot.services.auth_service import AuthService
from qna_bot.services.chat_service import ChatService
from qna_bot.services.document_service import DocumentService
from qna_bot.services.mcp_service import MCPService
from qna_bot.services.user_service import UserService
from qna_bot.ui.agent_workspace import render_agent_workspace_page
from qna_bot.ui.auth_views import render_account_page
from qna_bot.ui.mcp_dashboard import render_mcp_dashboard
from qna_bot.ui.mcp_marketplace import render_mcp_marketplace_page
from qna_bot.ui.workspace import (
    render_approval_center_page,
    render_knowledge_base_page,
    render_settings_page,
    render_tool_marketplace_page,
)


def dispatch_workspace_page(
    page: str,
    settings: AppSettings,
    sidebar_cfg: Any,
    auth_session: Any,
    auth_svc: AuthService,
    user_svc: UserService,
    chat_svc: ChatService,
    doc_svc: DocumentService,
    mcp_svc: MCPService,
    appr_svc: ApprovalService,
    vector_store: Any,
    on_signed_out: Callable[[], None],
    on_clear_chat: Callable[[], None],
    handle_approval_decision: Callable[[str, bool, ChatService, Any], None],
) -> bool:
    """Dispatch and render specialized workspace pages. Returns True if handled, False for Chat."""
    if page == "Agent Workspace":
        render_agent_workspace_page(
            user_id=sidebar_cfg.user_id,
            mcp_service=mcp_svc,
            document_service=doc_svc,
            approval_service=appr_svc,
            on_approval_decision=lambda aid, ok: handle_approval_decision(aid, ok, chat_svc, sidebar_cfg),
        )
        return True

    if page == "Knowledge Base":
        render_knowledge_base_page(settings, sidebar_cfg.user_id, doc_svc)
        return True

    if page == "MCP Marketplace":
        render_mcp_marketplace_page(settings, sidebar_cfg.user_id, mcp_svc)
        return True

    if page == "MCP Servers":
        render_mcp_dashboard(sidebar_cfg.user_id, mcp_svc)
        return True

    if page == "Tool Marketplace":
        render_tool_marketplace_page(sidebar_cfg.user_id, mcp_svc)
        return True

    if page == "Approvals":
        render_approval_center_page(
            user_id=sidebar_cfg.user_id,
            approval_service=appr_svc,
            chat_service=chat_svc,
            on_decision=lambda aid, ok: handle_approval_decision(aid, ok, chat_svc, sidebar_cfg),
        )
        return True

    if page == "Account":
        render_account_page(
            settings=settings,
            auth_session=auth_session,
            auth_service=auth_svc,
            user_service=user_svc,
            vector_store=vector_store,
            on_signed_out=on_signed_out,
        )
        return True

    if page == "Settings":
        render_settings_page(
            settings=settings,
            user_id=sidebar_cfg.user_id,
            on_clear_chat=on_clear_chat,
            user_service=user_svc,
            document_service=doc_svc,
        )
        return True

    return False
