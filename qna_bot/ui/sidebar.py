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
from qna_bot.ui.design_system import BRAND_SVG, render_html

WORKSPACE_PAGES = (
    "Chat",
    "Agent Workspace",
    "Knowledge Base",
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
    return str(st.session_state.get("active_page", "Chat"))


def _render_grouped_conversations(
    user_id: str,
    search_query: str,
    chat_service: Optional[ChatService],
    conv_service: Optional[ConversationService],
) -> None:
    render_html('<div class="sb-section-label">Conversations</div>')

    if conv_service is not None:
        try:
            grouped = conv_service.list_threads_grouped(user_id=user_id, search_query=search_query)
        except Exception as exc:
            st.caption(f"History unavailable: {exc}")
            grouped = {}
    elif chat_service is not None:
        try:
            items = [
                t
                for t in chat_service.list_threads(user_id=user_id, search_query=search_query)
                if t.get("message_count", 0) > 0
            ]
            grouped = {"Recent": items} if items else {}
        except Exception:
            grouped = {}
    else:
        recent = list(st.session_state.get("recent_chats") or [])
        if search_query:
            q = search_query.lower()
            recent = [c for c in recent if q in str(c.get("title", "")).lower()]
        grouped = {"Recent": recent} if recent else {}

    if not grouped:
        st.caption("No matching conversations." if search_query else "No saved conversations yet.")
        return

    active_thread_id = str(st.session_state.get("current_chat_id") or "")
    try:
        conv_box = st.container(height=250, border=False)
    except TypeError:
        conv_box = st.container()

    with conv_box:
        for bucket, threads in grouped.items():
            render_html(f'<div class="sb-bucket-label">{html.escape(bucket)}</div>')
            for chat in threads[:15]:
                tid = str(chat.get("id") or chat.get("thread_id", ""))
                full_title = str(chat.get("title", "Conversation") or "Conversation").strip()
                pinned = bool(chat.get("is_pinned", False))
                is_current = tid == active_thread_id

                col_open, col_menu = st.columns([5.2, 1])
                with col_open:
                    if st.button(
                        full_title,
                        key=f"restore_{tid}",
                        type="secondary" if is_current else "tertiary",
                        icon=":material/keep:" if pinned else None,
                        help=full_title,
                        use_container_width=True,
                    ):
                        if conv_service is not None:
                            msgs = conv_service.load_history(tid, user_id=user_id)
                            restore_chat({"id": tid, "messages": msgs})
                        elif chat_service is not None:
                            msgs = chat_service.load_thread_history(tid, user_id=user_id)
                            restore_chat({"id": tid, "messages": msgs})
                        else:
                            restore_chat(chat)
                        st.session_state.active_page = "Chat"
                        st.rerun()

                with col_menu:
                    if conv_service is not None:
                        with st.popover("", icon=":material/more_horiz:", help=f"Actions for {full_title}", use_container_width=True):
                            st.caption(full_title)
                            if st.button(
                                "Unpin" if pinned else "Pin",
                                key=f"pin_{tid}",
                                icon=":material/keep:",
                                use_container_width=True,
                            ):
                                conv_service.set_pinned(tid, user_id=user_id, pinned=not pinned)
                                st.rerun()

                            new_name = st.text_input(
                                "Rename conversation",
                                value=full_title,
                                key=f"rename_val_{tid}",
                            )
                            if st.button("Save Name", key=f"rename_btn_{tid}", use_container_width=True):
                                if new_name.strip():
                                    conv_service.rename_thread(tid, user_id=user_id, new_title=new_name.strip())
                                    st.rerun()

                            if st.session_state.get("delete_thread_id") != tid:
                                if st.button(
                                    "Delete",
                                    key=f"del_ask_{tid}",
                                    icon=":material/delete:",
                                    use_container_width=True,
                                ):
                                    st.session_state.delete_thread_id = tid
                                    st.rerun()
                            else:
                                st.warning("Delete permanently?")
                                c_yes, c_no = st.columns(2)
                                with c_yes:
                                    if st.button(
                                        "Delete",
                                        type="primary",
                                        key=f"del_confirm_{tid}",
                                        use_container_width=True,
                                    ):
                                        conv_service.delete_thread(tid, user_id=user_id)
                                        st.session_state.delete_thread_id = None
                                        if st.session_state.get("current_chat_id") == tid:
                                            st.session_state.messages = []
                                        st.rerun()
                                with c_no:
                                    if st.button("Cancel", key=f"del_cancel_{tid}", use_container_width=True):
                                        st.session_state.delete_thread_id = None
                                        st.rerun()


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
            f'<div class="sb-brand-left">{BRAND_SVG}<span class="sb-brand">QnAGPT</span></div>'
            f'<span class="sb-badge">Workspace</span>'
            f"</div>"
        )

        if authenticated_user is not None:
            with st.popover(f"{display_identity}", icon=":material/account_circle:", use_container_width=True):
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

        if st.button("New Chat", type="primary", icon=":material/add:", use_container_width=True):
            st.session_state.active_page = "Chat"
            on_new_chat()

        search_query = st.text_input(
            "Search Conversations",
            placeholder="Search conversations...",
            label_visibility="collapsed",
            key="sb_search_input",
        )
        st.session_state.search_query = search_query.strip()

        active_page = _render_navigation()

        render_html('<hr class="sb-divider"/>')
        _render_grouped_conversations(
            user_id=active_user_id,
            search_query=st.session_state.search_query,
            chat_service=chat_service,
            conv_service=conversation_service,
        )

        render_html('<hr class="sb-divider"/>')
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
            f'<div class="sb-user-pill">Account: <strong>{html.escape(display_identity)}</strong></div>'
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
