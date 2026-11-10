"""Chat message history and clean conversational turn rendering components."""

import html
from typing import Any, Callable, Dict, List, Optional, Sequence

import streamlit as st

from qna_bot.ui.components import render_copy_button, render_sources_panel
from qna_bot.ui.design_system import render_html


def render_user_message(
    content: str,
    image_b64: Optional[str] = None,
    image_mime: str = "image/png",
    images: Optional[Sequence[Dict[str, Any]]] = None,
    attached_docs: Optional[Sequence[str]] = None,
) -> None:
    """Render a subtle right-aligned user prompt with attachment chips and image previews."""
    _, col_user = st.columns([1, 5])
    with col_user:
        render_html('<div class="msg-label-user">You</div>')

        doc_names = list(attached_docs or [])
        if doc_names:
            chips = "".join(
                f'<span class="attach-chip">{html.escape(name)}</span>'
                for name in doc_names
            )
            render_html(f'<div class="attach-chip-row">{chips}</div>')

        img_list = list(images or [])
        if image_b64 and not img_list:
            img_list = [{"b64": image_b64, "mime": image_mime}]
        for img in img_list:
            b64_val = img.get("b64")
            mime_val = img.get("mime") or "image/png"
            if b64_val:
                img_tag = (
                    f'<img src="data:{mime_val};base64,{b64_val}" '
                    'style="max-width:100%;max-height:240px;border-radius:8px;'
                    'display:inline-block;margin-bottom:8px;border:1px solid #E2E8F0;" />'
                )
                render_html(f'<div style="text-align:right">{img_tag}</div>')

        if content.strip():
            with st.chat_message("user"):
                st.markdown(content)


def render_ai_message(
    content: str,
    index: int,
    is_last_ai: bool = False,
    sources: Optional[Sequence[Dict[str, Any]]] = None,
    web_sources: Optional[Sequence[Dict[str, Any]]] = None,
    tool_events: Optional[Sequence[Dict[str, Any]]] = None,
    execution_trace: Optional[Sequence[Dict[str, Any]]] = None,
    on_feedback: Optional[Callable[[str, str], None]] = None,
) -> None:
    """Render an assistant response using clean document typography and a functional toolbar."""
    col_ai, _ = st.columns([11, 1])
    with col_ai:
        render_html('<div class="msg-label-ai">QnAGPT</div>')
        with st.chat_message("assistant"):
            st.markdown(content)
            render_sources_panel(
                sources=sources,
                web_sources=web_sources,
                tool_events=tool_events,
                execution_trace=execution_trace,
            )

        feedback_map = st.session_state.get("feedback") or {}
        cur_fb = feedback_map.get(index)

        a_help, a_unhelp, a_copy, a_exp, a_regen, _ = st.columns([1.05, 1.2, 0.9, 1.0, 1.35, 3.5])
        with a_help:
            if st.button(
                "Helpful",
                key=f"helpful_{index}",
                type="secondary" if cur_fb == "helpful" else "tertiary",
                icon=":material/thumb_up:",
            ):
                st.session_state.feedback[index] = "helpful"
                if on_feedback is not None:
                    try:
                        on_feedback("up", content)
                    except Exception:
                        pass
                st.toast("Marked as helpful.")
                st.rerun()
        with a_unhelp:
            if st.button(
                "Not helpful",
                key=f"unhelpful_{index}",
                type="secondary" if cur_fb == "unhelpful" else "tertiary",
                icon=":material/thumb_down:",
            ):
                st.session_state.feedback[index] = "unhelpful"
                if on_feedback is not None:
                    try:
                        on_feedback("down", content)
                    except Exception:
                        pass
                st.toast("Feedback recorded.")
                st.rerun()
        with a_copy:
            render_copy_button(content, f"ai_{index}")
        with a_exp:
            if callable(getattr(st, "download_button", None)):
                st.download_button(
                    "Export",
                    data=content,
                    file_name=f"qnagpt_response_{index + 1}.md",
                    mime="text/markdown",
                    icon=":material/download:",
                    key=f"export_ai_{index}",
                )
        with a_regen:
            if is_last_ai and st.button(
                "Regenerate",
                key=f"regen_{index}",
                type="tertiary",
                icon=":material/refresh:",
            ):
                st.session_state.messages.pop()
                st.session_state.prompt_to_send = "__REGENERATE__"
                st.rerun()


def render_chat_history(
    messages: List[Dict[str, Any]],
    search_query: str = "",
    on_feedback: Optional[Callable[[str, str], None]] = None,
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
                images=msg.get("images"),
                attached_docs=msg.get("attached_docs"),
            )
        elif role == "assistant":
            render_ai_message(
                content=content,
                index=idx,
                is_last_ai=(idx == last_ai_idx),
                sources=msg.get("sources"),
                web_sources=msg.get("web_sources"),
                tool_events=msg.get("tool_events"),
                execution_trace=msg.get("execution_trace"),
                on_feedback=on_feedback,
            )
