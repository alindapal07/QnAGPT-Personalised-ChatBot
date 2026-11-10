"""Conversation export serialization helpers."""

import json
from typing import Any, Mapping, Sequence


def export_as_json(
    chat_id: str,
    messages: Sequence[Mapping[str, Any]],
) -> str:
    """Serialize conversation history to formatted JSON."""
    payload = {
        "chat_id": chat_id,
        "messages": list(messages),
    }
    return json.dumps(payload, indent=2)


def export_as_markdown(messages: Sequence[Mapping[str, Any]]) -> str:
    """Serialize conversation history to Markdown."""
    lines = ["# Chat Export\n"]
    for msg in messages:
        speaker = "User" if msg.get("role") == "user" else "Assistant"
        lines.append(f"### {speaker}\n{msg.get('content', '')}\n")
    return "\n".join(lines)


def export_as_text(messages: Sequence[Mapping[str, Any]]) -> str:
    """Serialize conversation history to plain text."""
    return "\n".join(
        f"{str(msg.get('role', 'unknown')).upper()}: {msg.get('content', '')}"
        for msg in messages
    )
