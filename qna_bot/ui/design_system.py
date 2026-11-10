"""Centralized enterprise Light-mode design tokens, SVG icons, and safe HTML/UI primitives."""

import html
import re
from typing import Any, Dict, List, Literal, Sequence, Tuple

import streamlit as st

COLORS: Dict[str, str] = {
    "bg_main": "#FFFFFF",
    "bg_secondary": "#F8FAFC",
    "bg_sidebar": "#F8FAFC",
    "text_primary": "#172033",
    "text_secondary": "#64748B",
    "accent": "#315EAB",
    "border": "#E2E8F0",
    "hover": "#F1F5F9",
    "success": "#15803D",
    "warning": "#B45309",
    "error": "#B91C1C",
}

ToneType = Literal["neutral", "ok", "warn", "err", "accent"]

BRAND_SVG = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" '
    'stroke="#315EAB" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<rect x="3" y="3" width="18" height="18" rx="4"/>'
    '<path d="M8 9h8M8 13h5M8 17h3"/></svg>'
)

_ICON_DONE_SVG = (
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" '
    'stroke="#15803D" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="12" cy="12" r="9"/><path d="m9 12 2 2 4-4"/></svg>'
)

_ICON_ACTIVE_SVG = (
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" '
    'stroke="#315EAB" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="3" fill="#315EAB"/></svg>'
)

_ICON_PENDING_SVG = (
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" '
    'stroke="#94A3B8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="12" cy="12" r="9"/></svg>'
)

_ICON_ERROR_SVG = (
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" '
    'stroke="#B91C1C" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="12" cy="12" r="9"/><path d="m15 9-6 6M9 9l6 6"/></svg>'
)


def compact_html(markup: str) -> str:
    """Strip indentation and blank lines so HTML never breaks CommonMark blocks."""
    lines = [line.strip() for line in (markup or "").splitlines() if line.strip()]
    return re.sub(r">\s+<", "><", " ".join(lines))


def render_html(markup: str, target: Any = None) -> None:
    """Render pure HTML safely via st.html (bypasses Markdown parser completely) with safe fallback."""
    cleaned = compact_html(markup)
    host = st if target is None else target
    if callable(getattr(host, "html", None)):
        host.html(cleaned)
    elif callable(getattr(st, "html", None)) and target is None:
        st.html(cleaned)
    else:
        host.markdown(cleaned, unsafe_allow_html=True)


def status_pill_html(label: str, tone: ToneType = "neutral") -> str:
    """Return consistent semantic status pill HTML using centralized light tokens."""
    cls_map = {
        "ok": "status-pill-ok",
        "warn": "status-pill-warn",
        "err": "status-pill-err",
        "accent": "status-pill-accent",
        "neutral": "status-pill-neutral",
    }
    cls = cls_map.get(tone, "status-pill-neutral")
    return f'<span class="{cls}">{html.escape(label)}</span>'


def render_page_header(title: str, subtitle: str) -> None:
    """Render standardized enterprise page title and subtitle without Markdown/HTML leakage."""
    render_html(
        f'<div class="ds-page-header">'
        f'<h1 class="ds-page-title">{html.escape(title)}</h1>'
        f'<p class="ds-page-subtitle">{html.escape(subtitle)}</p>'
        f"</div>"
    )


def render_section_header(title: str, subtitle: str = "") -> None:
    """Render aligned section header with optional right-aligned metadata."""
    sub_html = f'<span class="ds-section-sub">{html.escape(subtitle)}</span>' if subtitle else ""
    render_html(
        f'<div class="ds-section-header">'
        f'<span class="ds-section-title">{html.escape(title)}</span>'
        f"{sub_html}"
        f"</div>"
    )


def render_metric_strip(items: Sequence[Tuple[str, Any]]) -> None:
    """Render a responsive aligned metric card grid with tabular numbers."""
    cards = "".join(
        f'<div class="ds-metric-card">'
        f'<div class="ds-metric-label">{html.escape(str(label))}</div>'
        f'<div class="ds-metric-value">{html.escape(str(value))}</div>'
        f"</div>"
        for label, value in items
    )
    render_html(f'<div class="ds-metric-grid">{cards}</div>')


def render_empty_state(title: str, description: str) -> None:
    """Render clean SaaS empty state block without decorative graphics."""
    render_html(
        f'<div class="ds-empty-state">'
        f'<div class="ds-empty-title">{html.escape(title)}</div>'
        f'<div class="ds-empty-desc">{html.escape(description)}</div>'
        f"</div>"
    )


def render_execution_timeline_html(
    steps: Sequence[Dict[str, Any]],
    header: str = "Request processing",
) -> str:
    """Build compact timeline HTML driven strictly by real backend execution stages."""
    rows: List[str] = []
    for step in steps:
        status = str(step.get("state") or "completed")
        label = html.escape(str(step.get("label") or ""))
        detail = str(step.get("detail") or "").strip()
        detail_html = f'<span class="tl-detail">&mdash; {html.escape(detail)}</span>' if detail else ""

        if status == "in_progress":
            icon_svg = _ICON_ACTIVE_SVG
            lbl_cls = "tl-label-active"
        elif status == "failed":
            icon_svg = _ICON_ERROR_SVG
            lbl_cls = "tl-label-done"
        elif status == "pending":
            icon_svg = _ICON_PENDING_SVG
            lbl_cls = "tl-label-pending"
        else:
            icon_svg = _ICON_DONE_SVG
            lbl_cls = "tl-label-done"

        rows.append(
            f'<div class="tl-row"><span class="tl-icon">{icon_svg}</span>'
            f'<div><span class="{lbl_cls}">{label}</span>{detail_html}</div></div>'
        )

    return compact_html(
        f'<div class="tl-wrap">'
        f'<div class="tl-header">{html.escape(header)}</div>'
        f'{"".join(rows)}'
        f"</div>"
    )
