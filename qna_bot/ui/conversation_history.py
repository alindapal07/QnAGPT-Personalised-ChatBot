"""Enterprise Light-mode conversation history navigation, grouping, and contextual menu."""

from dataclasses import dataclass
import html
from typing import Any, Callable, Dict, List, Optional, Tuple

import streamlit as st

from qna_bot.services.chat_service import ChatService
from qna_bot.services.conversation_service import ConversationService
from qna_bot.services.session_service import restore_chat
from qna_bot.ui.design_system import render_html


# ── Dialog helpers ────────────────────────────────────────────────────────────

def _render_rename_dialog_body(
    thread_id: str,
    current_title: str,
    user_id: str,
    conv_service: Optional[ConversationService],
    chat_service: Optional[ChatService],
) -> None:
    render_html(
        '<div style="font-size:13.5px;color:#475569;margin-bottom:12px;line-height:1.4;">'
        'Enter a new title for this conversation.'
        '</div>'
    )
    new_title = st.text_input(
        "Conversation title",
        value=current_title,
        key=f"dialog_rename_input_{thread_id}",
        label_visibility="collapsed",
    )
    col_cancel, col_save = st.columns([1, 1])
    with col_cancel:
        if st.button("Cancel", key=f"dialog_rename_cancel_{thread_id}", use_container_width=True):
            st.session_state["active_conv_dialog"] = None
            st.rerun()
    with col_save:
        if st.button(
            "Save",
            type="primary",
            key=f"dialog_rename_save_{thread_id}",
            use_container_width=True,
        ):
            clean_title = (new_title or "").strip()
            if not clean_title:
                st.error("Title cannot be empty.")
                return
            if conv_service is not None:
                conv_service.rename_thread(thread_id, user_id=user_id, new_title=clean_title)
            elif chat_service is not None and hasattr(chat_service, "rename_thread"):
                try:
                    chat_service.rename_thread(thread_id, user_id=user_id, new_title=clean_title)
                except Exception:
                    pass
            st.session_state["active_conv_dialog"] = None
            st.rerun()


def _render_delete_dialog_body(
    thread_id: str,
    current_title: str,
    user_id: str,
    conv_service: Optional[ConversationService],
    chat_service: Optional[ChatService],
) -> None:
    render_html(
        '<div style="font-size:13.5px;color:#475569;margin-bottom:16px;line-height:1.5;">'
        'This conversation will be permanently removed. This action cannot be undone.'
        '</div>'
    )
    col_cancel, col_del = st.columns([1, 1.2])
    with col_cancel:
        if st.button("Cancel", key=f"dialog_del_cancel_{thread_id}", use_container_width=True):
            st.session_state["active_conv_dialog"] = None
            st.rerun()
    with col_del:
        if st.button(
            "Delete conversation",
            type="primary",
            key=f"dialog_del_confirm_{thread_id}",
            use_container_width=True,
        ):
            if conv_service is not None:
                conv_service.delete_thread(thread_id, user_id=user_id)
            elif chat_service is not None and hasattr(chat_service, "delete_thread"):
                try:
                    chat_service.delete_thread(thread_id, user_id=user_id)
                except Exception:
                    pass
            if st.session_state.get("current_chat_id") == thread_id:
                st.session_state["current_chat_id"] = None
                st.session_state["messages"] = []
            st.session_state["active_conv_dialog"] = None
            st.rerun()


_dialog_decorator = getattr(st, "dialog", None) or getattr(st, "experimental_dialog", None)

if _dialog_decorator is not None:
    @_dialog_decorator("Rename conversation")
    def RenameConversationDialog(
        thread_id: str,
        current_title: str,
        user_id: str,
        conv_service: Optional[ConversationService],
        chat_service: Optional[ChatService],
    ) -> None:
        _render_rename_dialog_body(thread_id, current_title, user_id, conv_service, chat_service)

    @_dialog_decorator("Delete conversation?")
    def DeleteConversationDialog(
        thread_id: str,
        current_title: str,
        user_id: str,
        conv_service: Optional[ConversationService],
        chat_service: Optional[ChatService],
    ) -> None:
        _render_delete_dialog_body(thread_id, current_title, user_id, conv_service, chat_service)
else:
    def RenameConversationDialog(
        thread_id: str,
        current_title: str,
        user_id: str,
        conv_service: Optional[ConversationService],
        chat_service: Optional[ChatService],
    ) -> None:
        with st.expander("Rename conversation", expanded=True):
            _render_rename_dialog_body(thread_id, current_title, user_id, conv_service, chat_service)

    def DeleteConversationDialog(
        thread_id: str,
        current_title: str,
        user_id: str,
        conv_service: Optional[ConversationService],
        chat_service: Optional[ChatService],
    ) -> None:
        with st.expander("Delete conversation?", expanded=True):
            _render_delete_dialog_body(thread_id, current_title, user_id, conv_service, chat_service)


# ── Context Menu & Row Components ─────────────────────────────────────────────

def ConversationContextMenu(
    thread_id: str,
    title: str,
    pinned: bool,
    user_id: str,
    conv_service: Optional[ConversationService],
    chat_service: Optional[ChatService],
) -> None:
    """Render the floating dropdown contextual action menu inside an st.popover."""
    with st.popover(
        "",
        icon=":material/more_horiz:",
        help="Conversation options",
        use_container_width=True,
    ):
        render_html(f'<div class="conv-menu-header">{html.escape(title)}</div>')

        # 1. Rename
        if st.button(
            "Rename",
            icon=":material/edit:",
            key=f"ctx_rename_{thread_id}",
            use_container_width=True,
        ):
            st.session_state["active_conv_dialog"] = ("rename", thread_id, title)
            st.rerun()

        # 2. Pin / Unpin
        pin_label = "Unpin" if pinned else "Pin"
        if st.button(
            pin_label,
            icon=":material/keep:",
            key=f"ctx_pin_{thread_id}",
            use_container_width=True,
        ):
            if conv_service is not None:
                conv_service.set_pinned(thread_id, user_id=user_id, pinned=not pinned)
            elif chat_service is not None and hasattr(chat_service, "set_pinned"):
                try:
                    chat_service.set_pinned(thread_id, user_id=user_id, pinned=not pinned)
                except Exception:
                    pass
            st.rerun()

        # 3. Archive
        if st.button(
            "Archive",
            icon=":material/archive:",
            key=f"ctx_archive_{thread_id}",
            use_container_width=True,
        ):
            if conv_service is not None:
                conv_service.set_archived(thread_id, user_id=user_id, archived=True)
            if st.session_state.get("current_chat_id") == thread_id:
                st.session_state["current_chat_id"] = None
                st.session_state["messages"] = []
            st.rerun()

        render_html('<hr class="conv-menu-divider"/>')

        # 4. Delete
        if st.button(
            "Delete",
            icon=":material/delete:",
            key=f"ctx_del_{thread_id}",
            use_container_width=True,
        ):
            st.session_state["active_conv_dialog"] = ("delete", thread_id, title)
            st.rerun()


def ConversationRow(
    chat: Dict[str, Any],
    active_thread_id: str,
    user_id: str,
    conv_service: Optional[ConversationService],
    chat_service: Optional[ChatService],
) -> None:
    """Render a single compact conversation row with selection button and contextual actions."""
    tid = str(chat.get("id") or chat.get("thread_id", ""))
    full_title = str(chat.get("title", "Conversation") or "Conversation").strip()
    pinned = bool(chat.get("is_pinned", False))
    is_current = tid == active_thread_id

    col_open, col_menu = st.columns([5.2, 1.0])
    with col_open:
        item_icon = ":material/keep:" if pinned else ":material/chat_bubble_outline:"
        if st.button(
            full_title,
            key=f"restore_{tid}",
            type="secondary" if is_current else "tertiary",
            icon=item_icon,
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
        ConversationContextMenu(
            thread_id=tid,
            title=full_title,
            pinned=pinned,
            user_id=user_id,
            conv_service=conv_service,
            chat_service=chat_service,
        )


def ConversationGroup(
    bucket_label: str,
    threads: List[Dict[str, Any]],
    active_thread_id: str,
    user_id: str,
    conv_service: Optional[ConversationService],
    chat_service: Optional[ChatService],
) -> None:
    """Render a bucket grouping of conversation rows."""
    render_html(f'<div class="sb-bucket-label">{html.escape(bucket_label)}</div>')
    for chat in threads[:15]:
        ConversationRow(
            chat=chat,
            active_thread_id=active_thread_id,
            user_id=user_id,
            conv_service=conv_service,
            chat_service=chat_service,
        )


def ConversationHistory(
    user_id: str,
    search_query: str,
    chat_service: Optional[ChatService],
    conv_service: Optional[ConversationService],
    on_clear_chat: Optional[Callable[[], None]] = None,
) -> None:
    """Render the full conversation history navigation, header, grouped rows, and dialogs."""
    # Active modal dialog if requested
    active_dialog = st.session_state.get("active_conv_dialog")
    if active_dialog:
        dialog_type, target_tid, target_title = active_dialog
        if dialog_type == "rename":
            RenameConversationDialog(
                thread_id=target_tid,
                current_title=target_title,
                user_id=user_id,
                conv_service=conv_service,
                chat_service=chat_service,
            )
        elif dialog_type == "delete":
            DeleteConversationDialog(
                thread_id=target_tid,
                current_title=target_title,
                user_id=user_id,
                conv_service=conv_service,
                chat_service=chat_service,
            )

    # Header with title and Clear All
    c_lbl, c_clr = st.columns([3.8, 2.2])
    with c_lbl:
        render_html('<div class="sb-section-label" style="margin:2px 0 2px 2px;">Recent conversations</div>')
    with c_clr:
        if on_clear_chat is not None and st.button(
            "Clear All",
            type="tertiary",
            key="sb_conv_clear_all",
            help="Clear messages in the current conversation",
        ):
            on_clear_chat()

    # Retrieve grouped conversations
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
        conv_box = st.container(height=300, border=False)
    except TypeError:
        conv_box = st.container()

    with conv_box:
        for bucket, threads in grouped.items():
            ConversationGroup(
                bucket_label=bucket,
                threads=threads,
                active_thread_id=active_thread_id,
                user_id=user_id,
                conv_service=conv_service,
                chat_service=chat_service,
            )


# Backward-compatible alias
render_conversation_history = ConversationHistory
