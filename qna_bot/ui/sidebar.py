"""Sidebar navigation, search, recent chats, model settings, and export controls."""

from dataclasses import dataclass
import html
from typing import Callable

import streamlit as st

from qna_bot.config.settings import AppSettings
from qna_bot.services.export_service import (
    export_as_json,
    export_as_markdown,
    export_as_text,
)
from qna_bot.services.session_service import restore_chat


@dataclass(frozen=True)
class SidebarConfig:
    """User-selected model and generation parameters from the sidebar."""

    text_model: str
    vision_model: str
    temperature: float
    max_tokens: int
    system_prompt: str


def _render_connection_status(api_key_present: bool) -> None:
    if api_key_present:
        st.markdown(
            '<div class="sb-status"><span class="dot-ok"></span>Connected</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="sb-status"><span class="dot-err"></span>API key missing</div>',
            unsafe_allow_html=True,
        )
        st.caption("Set `GROQ_API_KEY` in your `.env` file.")


def _render_recent_chats() -> None:
    if not st.session_state.recent_chats:
        return

    st.markdown('<div class="sb-section-label">Recent</div>', unsafe_allow_html=True)
    for chat in st.session_state.recent_chats[:6]:
        col_title, col_open = st.columns([5, 1])
        with col_title:
            safe_title = html.escape(chat["title"])
            st.markdown(f'<div class="sb-chat-title">{safe_title}</div>', unsafe_allow_html=True)
        with col_open:
            if st.button("Open", key=f"restore_{chat['id']}"):
                restore_chat(chat)
                st.rerun()


def _render_export_controls() -> None:
    messages = st.session_state.messages
    if not messages:
        return

    st.markdown('<div class="sb-section-label">Export</div>', unsafe_allow_html=True)
    json_data = export_as_json(st.session_state.current_chat_id, messages)
    md_data = export_as_markdown(messages)
    txt_data = export_as_text(messages)

    col_json, col_md, col_txt = st.columns(3)
    with col_json:
        st.download_button(
            "JSON",
            data=json_data,
            file_name="chat.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_md:
        st.download_button(
            "MD",
            data=md_data,
            file_name="chat.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col_txt:
        st.download_button(
            "TXT",
            data=txt_data,
            file_name="chat.txt",
            mime="text/plain",
            use_container_width=True,
        )


def _render_clear_controls(on_clear_chat: Callable[[], None]) -> None:
    st.markdown('<hr class="sb-divider"/>', unsafe_allow_html=True)
    if not st.session_state.confirm_clear:
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.confirm_clear = True
            st.rerun()
        return

    st.warning("Clear this conversation?")
    col_confirm, col_cancel = st.columns(2)
    with col_confirm:
        if st.button("Clear", type="primary", use_container_width=True):
            on_clear_chat()
    with col_cancel:
        if st.button("Cancel", use_container_width=True):
            st.session_state.confirm_clear = False
            st.rerun()


def render_sidebar(
    settings: AppSettings,
    api_key_present: bool,
    on_new_chat: Callable[[], None],
    on_clear_chat: Callable[[], None],
) -> SidebarConfig:
    """Render the sidebar and return the active model/generation configuration."""
    with st.sidebar:
        st.markdown('<div class="sb-brand">AI Assistant</div>', unsafe_allow_html=True)

        if st.button("+ New chat", use_container_width=True):
            on_new_chat()

        st.markdown('<hr class="sb-divider"/>', unsafe_allow_html=True)
        _render_connection_status(api_key_present)

        st.markdown('<div class="sb-section-label">Search</div>', unsafe_allow_html=True)
        search_query = st.text_input(
            "Search",
            placeholder="Search conversations...",
            label_visibility="collapsed",
            key="sb_search_input",
        )
        st.session_state.search_query = search_query.strip()

        _render_recent_chats()
        st.markdown('<hr class="sb-divider"/>', unsafe_allow_html=True)

        with st.expander("Settings", expanded=False):
            st.markdown('<div class="sb-section-label">Model</div>', unsafe_allow_html=True)
            selected_model = st.selectbox(
                "Text model",
                options=list(settings.available_text_models),
                index=0,
                label_visibility="collapsed",
            )
            st.caption(
                f"Vision model: `{settings.default_vision_model}` (auto-routed for images)"
            )

            st.markdown('<div class="sb-section-label">Temperature</div>', unsafe_allow_html=True)
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=settings.default_temperature,
                step=0.05,
                label_visibility="collapsed",
            )

            st.markdown('<div class="sb-section-label">Max tokens</div>', unsafe_allow_html=True)
            max_tokens = st.slider(
                "Max tokens",
                min_value=256,
                max_value=4096,
                value=settings.default_max_tokens,
                step=256,
                label_visibility="collapsed",
            )

            with st.expander("System prompt", expanded=False):
                system_prompt = st.text_area(
                    "System instructions",
                    value=settings.default_system_prompt,
                    height=90,
                    label_visibility="collapsed",
                )

        _render_export_controls()
        _render_clear_controls(on_clear_chat)

    return SidebarConfig(
        text_model=selected_model,
        vision_model=settings.default_vision_model,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )
