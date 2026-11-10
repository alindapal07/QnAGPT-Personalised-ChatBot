"""Fixed-bottom responsive chat composer with integrated image and document uploads."""

from dataclasses import dataclass, field
from typing import List, Tuple

import streamlit as st

from qna_bot.config.settings import AppSettings
from qna_bot.core.exceptions import ImageValidationError
from qna_bot.services.image_service import validate_and_encode_image


@dataclass(frozen=True)
class ComposerSubmission:
    """Normalized submission returned by render_chat_composer."""

    text: str
    images: List[dict] = field(default_factory=list)
    documents: List[Tuple[str, bytes]] = field(default_factory=list)
    doc_action: str = "Index into Knowledge Base"


def render_chat_composer(
    settings: AppSettings,
    active_model: str,
    disabled: bool = False,
) -> ComposerSubmission | None:
    """Render aligned composer controls (Agent mode, Attachment policy, Model badge) + native st.chat_input."""
    mode_options = [*settings.chat_modes, "Agent Workspace"]
    current_mode = st.session_state.get("chat_mode", mode_options[0])
    mode_idx = mode_options.index(current_mode) if current_mode in mode_options else 0

    c_mode, c_doc_act, c_model = st.columns([2, 2.2, 1.8])
    with c_mode:
        selected_mode = st.selectbox(
            "Agent Mode",
            options=mode_options,
            index=mode_idx,
            key="composer_mode_select",
            label_visibility="collapsed",
        )
        st.session_state.chat_mode = selected_mode

    with c_doc_act:
        doc_actions = ("Index into Knowledge Base", "Analyze in message")
        cur_act = st.session_state.get("doc_attach_action", doc_actions[0])
        act_idx = doc_actions.index(cur_act) if cur_act in doc_actions else 0
        doc_action = st.selectbox(
            "Attachment behavior",
            options=doc_actions,
            index=act_idx,
            key="composer_doc_action",
            label_visibility="collapsed",
        )
        st.session_state.doc_attach_action = doc_action

    with c_model:
        st.caption(f"Model: `{active_model}` | Enter to send")

    accepted_types = list(settings.allowed_image_types) + list(settings.allowed_document_types)
    nonce = st.session_state.get("composer_nonce", 0)

    raw_value = st.chat_input(
        "Ask anything, research the web, run MCP tools, or attach files...",
        key=f"main_chat_composer_{nonce}",
        accept_file="multiple",
        file_type=accepted_types,
        disabled=disabled,
        submit_mode="stop" if not disabled else "disable",
    )
    if raw_value is None:
        return None

    if isinstance(raw_value, str):
        text = raw_value.strip()
        uploaded_files = []
    else:
        text = str(getattr(raw_value, "text", "") or "").strip()
        uploaded_files = list(getattr(raw_value, "files", None) or [])

    if not text and not uploaded_files:
        return None

    images: List[dict] = []
    documents: List[Tuple[str, bytes]] = []
    image_exts = {ext.lower() for ext in settings.allowed_image_types}

    for up_file in uploaded_files:
        fname = str(getattr(up_file, "name", "attachment"))
        ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
        raw_bytes = up_file.getvalue()
        if ext in image_exts:
            try:
                images.append(
                    validate_and_encode_image(
                        file_bytes=raw_bytes,
                        filename=fname,
                        max_size_mb=settings.max_image_size_mb,
                        allowed_types=settings.allowed_image_types,
                    )
                )
            except ImageValidationError as exc:
                st.error(f"Image '{fname}' skipped: {exc}")
        else:
            documents.append((fname, raw_bytes))

    if not text and not images and not documents:
        return None

    return ComposerSubmission(
        text=text or ("Please analyze the attached image(s)." if images else "Please analyze the attached document(s)."),
        images=images,
        documents=documents,
        doc_action=doc_action,
    )
