"""Enterprise Light-mode collapsible sidebar navigation, conversation history manager, and footer."""

from dataclasses import dataclass
import html
import re
from typing import Callable, Optional, Tuple
import uuid

import streamlit as st

from qna_bot.config.settings import AppSettings
from qna_bot.schemas.auth import UserProfile
from qna_bot.services.chat_service import ChatService
from qna_bot.services.conversation_service import ConversationService
from qna_bot.services.document_service import DocumentService
from qna_bot.services.mcp_service import MCPService
from qna_bot.services.session_service import restore_chat
from qna_bot.ui.conversation_history import ConversationHistory
from qna_bot.ui.design_system import BRAND_AI_SPARK_SVG, BRAND_SVG, render_html

WORKSPACE_PAGES = (
    "Chat",
    "Agent Workspace",
    "Knowledge Base",
    "Approvals",
    "MCP Marketplace",
    "MCP Servers",
    "Tool Marketplace",
    "Account",
    "Settings",
)

NAV_GROUPS: Tuple[Tuple[str, Tuple[Tuple[str, str], ...]], ...] = (
    (
        "Workspace",
        (
            ("Chat", ":material/chat:"),
            ("Agent Workspace", ":material/smart_toy:"),
            ("Knowledge Base", ":material/library_books:"),
            ("Approvals", ":material/verified_user:"),
        ),
    ),
    (
        "Integrations",
        (
            ("MCP Marketplace", ":material/extension:"),
            ("MCP Servers", ":material/dns:"),
            ("Tool Marketplace", ":material/build:"),
        ),
    ),
    (
        "Preferences",
        (
            ("Account", ":material/person:"),
            ("Settings", ":material/settings:"),
        ),
    ),
)


@dataclass(frozen=True)
class SidebarConfig:
    """Active workspace navigation, model, mode, and generation parameters."""

    text_model: str
    vision_model: str
    temperature: float
    max_tokens: int
    system_prompt: str
    chat_mode: str = "Auto (Agentic)"
    user_id: str = "local_user"
    active_page: str = "Chat"


def _render_navigation() -> str:
    active = str(st.session_state.get("active_page", "Chat"))
    if active not in WORKSPACE_PAGES:
        active = "Chat"
        st.session_state["active_page"] = "Chat"

    for group_label, items in NAV_GROUPS:
        render_html(f'<div class="sb-section-label">{html.escape(group_label)}</div>')
        for page, icon in items:
            is_active = page == active
            if st.button(
                page,
                key=f"nav_page_{page}",
                type="secondary" if is_active else "tertiary",
                icon=icon,
                use_container_width=True,
            ):
                st.session_state.active_page = page
                st.rerun()
    return active


def _render_grouped_conversations(
    user_id: str,
    search_query: str,
    chat_service: Optional[ChatService],
    conv_service: Optional[ConversationService],
    on_clear_chat: Optional[Callable[[], None]] = None,
) -> None:
    ConversationHistory(
        user_id=user_id,
        search_query=search_query,
        chat_service=chat_service,
        conv_service=conv_service,
        on_clear_chat=on_clear_chat,
    )


def render_sidebar(
    settings: AppSettings,
    api_key_present: bool,
    on_new_chat: Callable[[], None],
    on_clear_chat: Callable[[], None],
    chat_service: Optional[ChatService] = None,
    document_service: Optional[DocumentService] = None,
    mcp_service: Optional[MCPService] = None,
    conversation_service: Optional[ConversationService] = None,
    authenticated_user: Optional[UserProfile] = None,
    on_sign_out: Optional[Callable[[], None]] = None,
) -> SidebarConfig:
    """Render the responsive Light-mode enterprise workspace sidebar."""
    active_user_id = (
        authenticated_user.id
        if authenticated_user is not None
        else str(st.session_state.get("user_id") or settings.default_user_id)
    )
    display_identity = (
        authenticated_user.full_name or authenticated_user.email
        if authenticated_user is not None
        else active_user_id
    )

    with st.sidebar:
        render_html(
            f'<div class="sb-brand-row">'
            f'<div class="sb-brand-left">{BRAND_AI_SPARK_SVG}<span class="sb-brand">QnAGPT</span></div>'
            f'<span class="sb-badge">Pro Workspace</span>'
            f"</div>"
        )

        col_new, col_find = st.columns([4.2, 1.2])
        with col_new:
            if st.button(
                "New chat",
                type="primary",
                icon=":material/add:",
                key="sb_new_chat_btn",
                use_container_width=True,
            ):
                st.session_state.active_page = "Chat"
                on_new_chat()
        with col_find:
            if st.button(
                "",
                type="secondary",
                icon=":material/search:",
                key="sb_toggle_search",
                help="Search conversations",
                use_container_width=True,
            ):
                st.session_state["show_sb_search"] = not st.session_state.get("show_sb_search", False)
                st.rerun()

        if st.session_state.get("show_sb_search") or st.session_state.get("search_query"):
            search_query = st.text_input(
                "Search Conversations",
                placeholder="Search conversations...",
                label_visibility="collapsed",
                key="sb_search_input",
            )
            st.session_state.search_query = search_query.strip()
        else:
            st.session_state.search_query = ""

        _render_grouped_conversations(
            user_id=active_user_id,
            search_query=st.session_state.search_query,
            chat_service=chat_service,
            conv_service=conversation_service,
            on_clear_chat=on_clear_chat,
        )

        render_html('<hr class="sb-divider"/>')
        active_page = _render_navigation()

        render_html('<hr class="sb-divider"/>')

        avatar_letter = (authenticated_user.full_name or authenticated_user.email or "U")[0].upper() if authenticated_user else "U"
        email_display = authenticated_user.email if authenticated_user else "Local Session"

        render_html(
            f'<div class="sb-profile-row">'
            f'<div class="sb-avatar-circle">{html.escape(avatar_letter)}</div>'
            f'<div class="sb-profile-text">'
            f'<div class="sb-profile-name">{html.escape(display_identity)}</div>'
            f'<div class="sb-profile-email">{html.escape(email_display)}</div>'
            f'</div>'
            f'</div>'
        )

        if authenticated_user is not None:
            with st.popover("Account & Sign Out", icon=":material/account_circle:", use_container_width=True):
                st.caption(f"Signed in as `{authenticated_user.email}`")
                c_acc, c_out = st.columns(2)
                with c_acc:
                    if st.button("Account", key="sb_acct_btn", use_container_width=True):
                        st.session_state.active_page = "Account"
                        st.rerun()
                with c_out:
                    if st.button("Sign Out", key="sb_signout_btn", use_container_width=True):
                        if on_sign_out is not None:
                            on_sign_out()
                        st.rerun()

        mcp_summary = ""
        if mcp_service is not None:
            try:
                statuses = mcp_service.list_statuses(user_id=active_user_id)
                connected_cnt = sum(1 for s in statuses if s.connected)
                mcp_summary = f"MCP: {connected_cnt}/{len(statuses)} connected"
            except Exception:
                mcp_summary = ""

        doc_summary = ""
        if document_service is not None:
            try:
                doc_cnt = len(document_service.list_documents(user_id=active_user_id))
                doc_summary = f"Docs: {doc_cnt} indexed"
            except Exception:
                doc_summary = ""

        dot_cls = "dot-ok" if api_key_present else "dot-err"
        status_label = "AI Runtime Connected" if api_key_present else "GROQ_API_KEY missing"
        meta_parts = " &middot; ".join(p for p in (mcp_summary, doc_summary) if p)
        meta_html = f'<div class="sb-meta-row">{meta_parts}</div>' if meta_parts else ""

        render_html(
            f'<div class="sb-footer-card">'
            f'<div class="sb-status"><span class="{dot_cls}"></span>{html.escape(status_label)}</div>'
            f"{meta_html}"
            f"</div>"
        )

    selected_model = str(st.session_state.get("cfg_text_model") or settings.default_text_model)
    if selected_model not in settings.available_text_models:
        selected_model = settings.default_text_model

    return SidebarConfig(
        text_model=selected_model,
        vision_model=settings.default_vision_model,
        temperature=float(st.session_state.get("cfg_temperature", settings.default_temperature)),
        max_tokens=int(st.session_state.get("cfg_max_tokens", settings.default_max_tokens)),
        system_prompt=str(st.session_state.get("cfg_system_prompt", settings.default_system_prompt)),
        chat_mode=str(st.session_state.get("chat_mode", settings.chat_modes[0])),
        user_id=active_user_id,
        active_page=active_page,
    )
