"""Chat message history and turn rendering components."""

from typing import Any, Dict, List, Optional

import streamlit as st

from qna_bot.ui.components import render_copy_button


def render_user_message(
    content: str,
    image_b64: Optional[str] = None,
    image_mime: str = "image/png",
) -> None:
    """Render a user message aligned to the right side."""
    _, col_user = st.columns([1, 4])
    with col_user:
        st.markdown('<div class="msg-label-user">You</div>', unsafe_allow_html=True)
        if image_b64:
            img_tag = (
                f'<img src="data:{image_mime};base64,{image_b64}" '
                'style="max-width:100%;max-height:260px;border-radius:6px;'
                'display:block;margin-bottom:8px;border:1px solid var(--border);" />'
            )
            st.markdown(f'<div style="text-align:right">{img_tag}</div>', unsafe_allow_html=True)
        if content.strip():
            with st.chat_message("user"):
                st.markdown(content)


def render_ai_message(
    content: str,
    index: int,
    is_last_ai: bool = False,
) -> None:
    """Render an assistant message aligned to the left side with action controls."""
    col_ai, _ = st.columns([4, 1])
    with col_ai:
        st.markdown('<div class="msg-label-ai">Assistant</div>', unsafe_allow_html=True)
        with st.chat_message("assistant"):
            st.markdown(content)

        a1, a2, a3, a4, _ = st.columns([1, 1.2, 1, 1.4, 4])
        with a1:
            if st.button("Helpful", key=f"helpful_{index}"):
                st.session_state.feedback[index] = "helpful"
                st.toast("Marked as helpful.")
        with a2:
            if st.button("Not helpful", key=f"unhelpful_{index}"):
                st.session_state.feedback[index] = "unhelpful"
                st.toast("Feedback recorded.")
        with a3:
            render_copy_button(content, f"ai_{index}")
        with a4:
            if is_last_ai and st.button("Regenerate", key=f"regen_{index}"):
                st.session_state.messages.pop()
                st.session_state.prompt_to_send = "__REGENERATE__"
                st.rerun()


def render_chat_history(
    messages: List[Dict[str, Any]],
    search_query: str = "",
) -> None:
    """Render conversation history filtered by an optional search query."""
    ai_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    last_ai_idx = ai_indices[-1] if ai_indices else -1
    normalized_query = search_query.lower()

    for idx, msg in enumerate(messages):
        content = msg.get("content", "")
        if normalized_query and normalized_query not in content.lower():
            continue

        role = msg.get("role")
        if role == "user":
            render_user_message(
                content=content,
                image_b64=msg.get("image_b64"),
                image_mime=msg.get("image_mime", "image/png"),
            )
        elif role == "assistant":
            render_ai_message(
                content=content,
                index=idx,
                is_last_ai=(idx == last_ai_idx),
            )
