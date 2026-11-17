"""Session state lifecycle and workspace state management."""

import time
from typing import Any, Dict, Mapping, MutableMapping
import uuid

import streamlit as st

from qna_bot.config.settings import get_settings

_TITLE_MAX_LEN = 120


def _resolve_state(
    state: MutableMapping[str, Any] | None,
) -> MutableMapping[str, Any]:
    return st.session_state if state is None else state


def initialize_session_state(
    state: MutableMapping[str, Any] | None = None,
) -> None:
    """Populate missing Streamlit session state keys with default values."""
    target = _resolve_state(state)
    settings = get_settings()
    defaults: Dict[str, Any] = {
        "messages": [],
        "current_chat_id": f"conversation_{uuid.uuid4().hex[:12]}",
        "recent_chats": [],
        "pending_image": None,
        "feedback": {},
        "confirm_clear": False,
        "search_query": "",
        "prompt_to_send": None,
        "uploader_nonce": 0,
        "doc_uploader_nonce": 0,
        "composer_nonce": 0,
        "user_id": settings.default_user_id,
        "chat_mode": settings.chat_modes[0],
        "active_page": "Chat",
        "pending_approval": None,
        "agent_workspace_state": {
            "task": "",
            "status": "Idle",
            "planned_steps": [],
            "tool_executions": [],
            "final_result": "",
        },
        "doc_attach_action": "Index into Knowledge Base",
        "rename_thread_id": None,
        "delete_thread_id": None,
        "mkt_selected_connector": None,
        "mkt_cursor_stack": [],
    }
    for key, value in defaults.items():
        if key not in target:
            target[key] = value


def _derive_chat_title(messages: list[Dict[str, Any]]) -> str:
    raw_content = next(
        (m.get("content", "") for m in messages if m.get("role") == "user"),
        "",
    )
    if isinstance(raw_content, list):
        text_parts = [
            str(item.get("text", "")).strip()
            for item in raw_content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        has_image = any(
            isinstance(item, dict) and ("image" in str(item.get("type", "")) or "image_url" in item)
            for item in raw_content
        )
        text = " ".join(tp for tp in text_parts if tp)
        first_user = text or ("Image Analysis" if has_image else "Conversation")
    elif isinstance(raw_content, str):
        first_user = " ".join(raw_content.split())
    else:
        first_user = " ".join(str(raw_content or "").split())

    if first_user and len(first_user) > _TITLE_MAX_LEN:
        return f"{first_user[:_TITLE_MAX_LEN]}..."
    return first_user or "Conversation"


def save_current_chat_to_recent(
    state: MutableMapping[str, Any] | None = None,
) -> None:
    """Persist the active conversation into recent_chats if it has messages."""
    target = _resolve_state(state)
    messages = target.get("messages") or []
    if not messages:
        return

    chat_id = target.get("current_chat_id")
    recent_chats = target.setdefault("recent_chats", [])
    snapshot = {
        "id": chat_id,
        "title": _derive_chat_title(messages),
        "messages": list(messages),
        "timestamp": time.strftime("%b %d, %H:%M"),
    }

    for idx, existing in enumerate(recent_chats):
        if existing.get("id") == chat_id:
            recent_chats[idx] = snapshot
            return

    recent_chats.insert(0, snapshot)


def start_new_chat(state: MutableMapping[str, Any] | None = None) -> None:
    """Archive the active conversation and reset state for a fresh chat."""
    target = _resolve_state(state)
    save_current_chat_to_recent(target)
    target["messages"] = []
    target["pending_image"] = None
    target["pending_approval"] = None
    target["feedback"] = {}
    target["current_chat_id"] = f"conversation_{uuid.uuid4().hex[:12]}"
    target["confirm_clear"] = False
    target["uploader_nonce"] = int(target.get("uploader_nonce", 0)) + 1
    target["composer_nonce"] = int(target.get("composer_nonce", 0)) + 1


def clear_current_chat(state: MutableMapping[str, Any] | None = None) -> None:
    """Clear the active conversation without creating a new recent entry."""
    target = _resolve_state(state)
    target["messages"] = []
    target["pending_image"] = None
    target["pending_approval"] = None
    target["feedback"] = {}
    target["confirm_clear"] = False
    target["uploader_nonce"] = int(target.get("uploader_nonce", 0)) + 1
    target["composer_nonce"] = int(target.get("composer_nonce", 0)) + 1


def restore_chat(
    chat: Mapping[str, Any],
    state: MutableMapping[str, Any] | None = None,
) -> None:
    """Load a saved conversation from recent_chats into the active view."""
    target = _resolve_state(state)
    target["messages"] = list(chat.get("messages", []))
    target["current_chat_id"] = chat.get("id", f"conversation_{uuid.uuid4().hex[:12]}")
    target["pending_image"] = None
    target["pending_approval"] = None
    target["feedback"] = {}
    target["confirm_clear"] = False
    target["uploader_nonce"] = int(target.get("uploader_nonce", 0)) + 1
    target["composer_nonce"] = int(target.get("composer_nonce", 0)) + 1
