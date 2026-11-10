"""Sidebar navigation, chat mode selector, document ingestion, persistent threads, and settings."""

from dataclasses import dataclass
import html
import re
from typing import Any, Callable, Dict, List, Optional
import uuid

import streamlit as st

from qna_bot.config.settings import AppSettings
from qna_bot.services.chat_service import ChatService
from qna_bot.services.document_service import DocumentService
from qna_bot.services.export_service import (
    export_as_json,
    export_as_markdown,
    export_as_text,
)
from qna_bot.services.session_service import restore_chat


@dataclass(frozen=True)
class SidebarConfig:
    """User-selected model, chat mode, workspace, and generation parameters."""

    text_model: str
    vision_model: str
    temperature: float
    max_tokens: int
    system_prompt: str
    chat_mode: str = "Auto (Agentic)"
    user_id: str = "local_user"


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


def _render_document_manager(
    settings: AppSettings,
    user_id: str,
    document_service: Optional[DocumentService],
) -> None:
    if document_service is None:
        return

    st.markdown('<div class="sb-section-label">Knowledge Base</div>', unsafe_allow_html=True)
    nonce = st.session_state.get("doc_uploader_nonce", 0)
    uploaded_docs = st.file_uploader(
        "Upload documents",
        type=list(settings.allowed_document_types),
        accept_multiple_files=True,
        key=f"doc_file_uploader_{nonce}",
        label_visibility="collapsed",
    )

    if uploaded_docs:
        if st.button("Ingest Documents", use_container_width=True, key="btn_ingest_docs"):
            payload = [(f.name, f.getvalue()) for f in uploaded_docs]
            with st.spinner("Indexing documents..."):
                results = document_service.ingest_files(payload, user_id=user_id)
            st.session_state.doc_uploader_nonce = nonce + 1
            feedback_items = []
            for res in results:
                if res.status == "ingested":
                    feedback_items.append(("success", f"Indexed {res.filename} ({res.chunk_count} chunks)"))
                elif res.status == "duplicate":
                    feedback_items.append(("info", f"Skipped duplicate: {res.filename}"))
                else:
                    feedback_items.append(("error", f"Failed {res.filename}: {res.error}"))
            st.session_state["doc_ingest_feedback"] = feedback_items
            st.rerun()

    for level, msg in st.session_state.pop("doc_ingest_feedback", []):
        getattr(st, level)(msg)

    try:
        indexed_docs = document_service.list_documents(user_id=user_id)
    except Exception as exc:
        st.error(f"Document index error: {exc}")
        indexed_docs = []

    if indexed_docs:
        st.markdown('<div class="sb-section-label">Indexed Documents</div>', unsafe_allow_html=True)
        for doc in indexed_docs:
            doc_id = str(doc.get("document_id", ""))
            fname = html.escape(str(doc.get("filename", "Document")))
            chunks = int(doc.get("chunk_count", 0))
            col_name, col_del = st.columns([4, 1])
            with col_name:
                st.markdown(
                    f'<div class="sb-doc-name" title="{fname}">{fname}</div>'
                    f'<div class="sb-doc-meta">{chunks} chunks</div>',
                    unsafe_allow_html=True,
                )
            with col_del:
                if st.button("Del", key=f"del_doc_{doc_id}", help=f"Remove {doc.get('filename')}"):
                    document_service.remove_document(doc_id, user_id=user_id)
                    st.rerun()

        if st.button("Clear Document Index", use_container_width=True, key="btn_clear_doc_index"):
            document_service.clear_index(user_id=user_id)
            st.rerun()


def _render_conversation_threads(
    user_id: str,
    chat_service: Optional[ChatService],
) -> None:
    threads: List[Dict[str, Any]] = []
    if chat_service is not None:
        try:
            threads = [
                t
                for t in chat_service.list_threads(user_id=user_id)
                if t.get("message_count", 0) > 0
            ]
        except Exception as exc:
            st.caption(f"History unavailable: {exc}")
    elif st.session_state.recent_chats:
        threads = list(st.session_state.recent_chats)

    if not threads:
        return

    st.markdown('<div class="sb-section-label">Conversation History</div>', unsafe_allow_html=True)
    for chat in threads[:8]:
        tid = str(chat.get("id") or chat.get("thread_id", ""))
        col_title, col_open = st.columns([5, 1])
        with col_title:
            safe_title = html.escape(str(chat.get("title", "Conversation")))
            st.markdown(f'<div class="sb-chat-title">{safe_title}</div>', unsafe_allow_html=True)
        with col_open:
            if st.button("Open", key=f"restore_{tid}"):
                if chat_service is not None:
                    try:
                        msgs = chat_service.load_thread_history(tid, user_id=user_id)
                        restore_chat({"id": tid, "messages": msgs})
                    except Exception as exc:
                        st.error(f"Could not open thread: {exc}")
                        return
                else:
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
    chat_service: Optional[ChatService] = None,
    document_service: Optional[DocumentService] = None,
) -> SidebarConfig:
    """Render the sidebar and return the active model, mode, and generation configuration."""
    with st.sidebar:
        st.markdown('<div class="sb-brand">QnAGPT Assistant</div>', unsafe_allow_html=True)

        if st.button("+ New chat", use_container_width=True):
            on_new_chat()

        st.markdown('<hr class="sb-divider"/>', unsafe_allow_html=True)
        _render_connection_status(api_key_present)

        st.markdown('<div class="sb-section-label">Chat Mode</div>', unsafe_allow_html=True)
        mode_options = list(settings.chat_modes)
        current_mode = st.session_state.get("chat_mode", mode_options[0])
        mode_idx = mode_options.index(current_mode) if current_mode in mode_options else 0
        chat_mode = st.selectbox(
            "Chat Mode",
            options=mode_options,
            index=mode_idx,
            label_visibility="collapsed",
            key="sb_chat_mode_select",
        )
        st.session_state.chat_mode = chat_mode

        active_user_id = str(st.session_state.get("user_id") or settings.default_user_id)
        _render_document_manager(settings, active_user_id, document_service)

        st.markdown('<div class="sb-section-label">Search</div>', unsafe_allow_html=True)
        search_query = st.text_input(
            "Search",
            placeholder="Search conversations...",
            label_visibility="collapsed",
            key="sb_search_input",
        )
        st.session_state.search_query = search_query.strip()

        _render_conversation_threads(active_user_id, chat_service)
        st.markdown('<hr class="sb-divider"/>', unsafe_allow_html=True)

        with st.expander("Settings", expanded=False):
            st.markdown('<div class="sb-section-label">Workspace ID</div>', unsafe_allow_html=True)
            raw_user = st.text_input(
                "Workspace ID",
                value=active_user_id,
                help="Isolates uploaded documents and conversation threads per user workspace.",
                label_visibility="collapsed",
                key="sb_workspace_user_id",
            )
            cleaned_user = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw_user.strip()).strip("_")
            resolved_user = cleaned_user or settings.default_user_id
            if resolved_user != active_user_id:
                st.session_state.user_id = resolved_user
                st.session_state.messages = []
                st.session_state.current_chat_id = f"conversation_{uuid.uuid4().hex[:12]}"
                st.session_state.pending_image = None
                st.session_state.feedback = {}
                st.session_state.confirm_clear = False
                st.rerun()

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
        chat_mode=chat_mode,
        user_id=active_user_id,
    )
