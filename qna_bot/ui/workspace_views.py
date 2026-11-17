"""Backward-compatibility facade for workspace views."""

from qna_bot.ui.workspace import (
    BUILTIN_TOOL_SPECS,
    _fmt_size,
    _fmt_ts,
    render_knowledge_base_page,
    render_settings_page,
    render_tool_marketplace_page,
)

__all__ = [
    "BUILTIN_TOOL_SPECS",
    "_fmt_size",
    "_fmt_ts",
    "render_knowledge_base_page",
    "render_settings_page",
    "render_tool_marketplace_page",
]
