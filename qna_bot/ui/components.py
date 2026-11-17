"""Reusable Light-mode Streamlit UI components (header, dynamic suggestions, copy, sources, HITL approval)."""

import html
import json
import os
from typing import Any, Callable, Dict, Optional, Sequence
from urllib.parse import urlparse

import streamlit as st

from qna_bot.config.settings import AppSettings
from qna_bot.services.suggestion_service import SuggestedPrompt
from qna_bot.ui.design_system import render_execution_timeline_html, render_html, status_pill_html

DEFAULT_SUGGESTIONS: Sequence[SuggestedPrompt] = (
    SuggestedPrompt(
        label="Explain a technical concept",
        prompt="Explain how transformer self-attention and retrieval-augmented generation work together in modern AI architectures.",
        category="Concepts",
        icon=":material/lightbulb:",
    ),
    SuggestedPrompt(
        label="Summarize a document",
        prompt="Analyze and summarize the core insights from my uploaded Knowledge Base documents with citations.",
        category="Documents",
        icon=":material/library_books:",
    ),
    SuggestedPrompt(
        label="Analyze research material",
        prompt="Conduct an objective research synthesis comparing recent agentic workflow patterns and tool integration models.",
        category="Research",
        icon=":material/search:",
    ),
    SuggestedPrompt(
        label="Search my knowledge base",
        prompt="Search my indexed knowledge base for references to security, architecture, and data retention policies.",
        category="Knowledge",
        icon=":material/find_in_page:",
    ),
    SuggestedPrompt(
        label="Help me solve a problem",
        prompt="Break down an engineering challenge into clear, step-by-step diagnostic and implementation phases.",
        category="Problem Solving",
        icon=":material/build:",
    ),
)


def render_copy_button(text: str, button_id: str) -> None:
    """Render a Light-theme JS-powered copy button that displays 'Copied' for 1.8s."""
    safe_json = json.dumps(text).replace("</", "<\\/")
    html_code = (
        '<style>html, body { margin: 0; padding: 0; overflow: hidden; background: transparent; }</style>'
        '<div style="margin: 0; padding: 0;">'
        f'<button id="btn_{button_id}" onclick="copyText_{button_id}()" style="'
        'background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; color: #172033; '
        'font-family: Inter, system-ui, sans-serif; font-size: 12px; font-weight: 500; '
        'padding: 3px 10px; cursor: pointer; line-height: 1.45;">Copy</button>'
        "<script>"
        f"function copyText_{button_id}() {{"
        f"const val = {safe_json};"
        f'const btn = document.getElementById("btn_{button_id}");'
        "if (navigator.clipboard && navigator.clipboard.writeText) {"
        "navigator.clipboard.writeText(val).then(() => {"
        'btn.innerText = "Copied"; btn.style.borderColor = "#315EAB"; btn.style.color = "#315EAB";'
        f'setTimeout(() => {{ btn.innerText = "Copy"; btn.style.borderColor = "#E2E8F0"; btn.style.color = "#172033"; }}, 1800);'
        f"}}).catch(() => fallback_{button_id}(val, btn));"
        f"}} else {{ fallback_{button_id}(val, btn); }}"
        "}"
        f"function fallback_{button_id}(val, btn) {{"
        'const ta = document.createElement("textarea"); ta.value = val; ta.style.position = "fixed"; ta.style.opacity = "0";'
        'document.body.appendChild(ta); ta.select(); document.execCommand("copy"); document.body.removeChild(ta);'
        'btn.innerText = "Copied"; setTimeout(() => { btn.innerText = "Copy"; }, 1800);'
        "}"
        "</script></div>"
    )
    if hasattr(st, "iframe"):
        st.iframe(html_code, height=30, alt="Copy assistant response")
    else:
        import streamlit.components.v1 as components

        components.html(html_code, height=30)


def render_header(
    model_name: str,
    chat_mode: Optional[str] = None,
    page_title: str = "Chat",
    conversation_title: Optional[str] = None,
    settings: Optional[AppSettings] = None,
    on_clear_chat: Optional[Callable[[], None]] = None,
) -> None:
    """Render compact Chat header with active conversation title, model, mode, and contextual popover."""
    escaped_model = html.escape(model_name)
    escaped_page = html.escape(page_title)
    sub_html = (
        f'<span class="app-header-sub" title="{html.escape(conversation_title)}">/ {html.escape(conversation_title)}</span>'
        if conversation_title and page_title == "Chat"
        else ""
    )
    mode_html = (
        f'<span class="app-header-pill">{html.escape(chat_mode)}</span>'
        if chat_mode
        else ""
    )

    show_actions = page_title == "Chat" and settings is not None and callable(getattr(st, "popover", None))
    if show_actions:
        col_head, col_cfg = st.columns([5.3, 1.3])
    else:
        col_head = st.container()
        col_cfg = None

    with col_head:
        render_html(
            f'<div class="app-header">'
            f'<div class="app-header-left"><span class="app-header-title">QnAGPT &middot; {escaped_page}</span>{sub_html}</div>'
            f'<div class="app-header-meta">{mode_html}<span class="app-header-pill">{escaped_model}</span></div>'
            f"</div>"
        )

    if col_cfg is not None and settings is not None:
        with col_cfg:
            with st.popover("Configure", icon=":material/settings:", use_container_width=True):
                mode_opts = [*settings.chat_modes, "Agent Workspace"]
                cur_mode = st.session_state.get("chat_mode", mode_opts[0])
                m_idx = mode_opts.index(cur_mode) if cur_mode in mode_opts else 0
                st.session_state["chat_mode"] = st.selectbox(
                    "Agent Mode",
                    options=mode_opts,
                    index=m_idx,
                    key="hdr_mode_select",
                )
                cur_model = str(st.session_state.get("cfg_text_model") or settings.default_text_model)
                mod_opts = list(settings.available_text_models)
                mod_idx = mod_opts.index(cur_model) if cur_model in mod_opts else 0
                st.session_state["cfg_text_model"] = st.selectbox(
                    "Model",
                    options=mod_opts,
                    index=mod_idx,
                    key="hdr_model_select",
                )
                if on_clear_chat is not None and st.session_state.get("messages"):
                    st.divider()
                    if st.button(
                        "Clear Conversation",
                        icon=":material/delete:",
                        key="hdr_clear_chat",
                        use_container_width=True,
                    ):
                        on_clear_chat()


def render_welcome_screen(
    suggestions: Optional[Sequence[SuggestedPrompt]] = None,
    on_refresh: Optional[Callable[[], None]] = None,
) -> None:
    """Render clean SaaS Chat landing view with dynamic workspace-aware starter prompts."""
    render_html(
        '<div class="welcome-wrap">'
        '<div class="welcome-title">What would you like to explore today?</div>'
        '<div class="welcome-sub">Ask questions, explore your documents, or let an AI agent help you get things done.</div>'
        "</div>"
    )

    active_suggestions = list(suggestions or DEFAULT_SUGGESTIONS)
    head_col, ref_col = st.columns([5, 1])
    with head_col:
        render_html('<div class="sug-header-label">Suggested for this workspace</div>')
    with ref_col:
        if on_refresh is not None and st.button(
            "Refresh",
            key="sug_refresh_btn",
            type="tertiary",
            icon=":material/refresh:",
            help="Refresh contextual prompt suggestions",
            use_container_width=True,
        ):
            on_refresh()

    cols = st.columns(2)
    for idx, item in enumerate(active_suggestions):
        if isinstance(item, tuple):
            label, prompt = item[0], item[1]
            icon = item[2] if len(item) > 2 else ":material/search:"
        else:
            label = item.label
            prompt = item.prompt
            icon = item.icon
        with cols[idx % 2]:
            if st.button(label, key=f"sug_{idx}", icon=icon, help=prompt, use_container_width=True):
                st.session_state.prompt_to_send = prompt
                st.rerun()


def render_sources_panel(
    sources: Optional[Sequence[Dict[str, Any]]] = None,
    web_sources: Optional[Sequence[Dict[str, Any]]] = None,
    tool_events: Optional[Sequence[Dict[str, Any]]] = None,
    execution_trace: Optional[Sequence[Dict[str, Any]]] = None,
) -> None:
    """Render compact document citations, web sources with publisher links, and tool execution trace."""
    doc_list = list(sources or [])
    web_list = list(web_sources or [])
    tool_list = list(tool_events or [])
    trace_list = list(execution_trace or [])
    if not doc_list and not web_list and not tool_list and not trace_list:
        return

    if doc_list or web_list:
        total = len(doc_list) + len(web_list)
        with st.expander(f"Sources ({total})", expanded=False):
            if doc_list:
                st.markdown("**Knowledge Base References**")
                for idx, src in enumerate(doc_list, start=1):
                    filename = html.escape(str(src.get("source", "Document")))
                    page = src.get("page")
                    page_text = f" &middot; Page {int(page)}" if page is not None else ""
                    excerpt = html.escape(str(src.get("excerpt", "")))
                    excerpt_html = f'<div class="source-card-snippet">{excerpt}</div>' if excerpt else ""
                    render_html(
                        f'<div class="source-card">'
                        f'<div class="source-card-meta"><span>{idx}. {filename}{page_text}</span>{status_pill_html("Indexed Doc", "accent")}</div>'
                        f"{excerpt_html}</div>"
                    )

            if web_list:
                st.markdown("**Web Sources**")
                for idx, web in enumerate(web_list, start=1):
                    title = html.escape(str(web.get("title", "Web Result")))
                    url = str(web.get("url", "")).strip()
                    host = (urlparse(url).netloc or "").replace("www.", "") if url else ""
                    snippet = html.escape(str(web.get("snippet", "")))
                    link_html = (
                        f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{title}</a>'
                        if url.startswith(("http://", "https://"))
                        else title
                    )
                    status_val = str(web.get("status", "")).strip()
                    if status_val == "read":
                        badge_html = status_pill_html("Page read", "ok")
                    elif status_val == "inaccessible":
                        badge_html = status_pill_html("Inaccessible", "warn")
                    elif status_val == "snippet_only":
                        badge_html = status_pill_html("Snippet", "neutral")
                    else:
                        badge_html = ""
                    domain_html = (
                        f'<span class="source-card-domain">{html.escape(host)}</span>' if host else ""
                    )
                    snippet_html = f'<div class="source-card-snippet">{snippet}</div>' if snippet else ""
                    render_html(
                        f'<div class="source-card">'
                        f'<div class="source-card-meta"><span>{idx}. {link_html} {domain_html}</span>{badge_html}</div>'
                        f"{snippet_html}</div>"
                    )

    if tool_list or trace_list:
        step_count = len(trace_list) if trace_list else len(tool_list)
        with st.expander(f"Execution & Tool Activity ({step_count})", expanded=False):
            if trace_list:
                render_html(render_execution_timeline_html(trace_list, header="Execution stages"))
            if tool_list:
                st.markdown("**Tool Calls**")
                for idx, ev in enumerate(tool_list, start=1):
                    srv = html.escape(str(ev.get("server_name", "Tool")))
                    tname = html.escape(str(ev.get("tool_name", "")))
                    ev_status = str(ev.get("status", "completed")).lower()
                    tone = "ok" if ev_status == "completed" else ("err" if ev_status in ("failed", "blocked", "rejected") else "warn")
                    dur = ev.get("duration_ms")
                    dur_txt = f" &middot; {dur:.0f} ms" if isinstance(dur, (int, float)) else ""
                    summary_raw = str(ev.get("summary", "") or "")
                    summary = html.escape(summary_raw[:260])
                    render_html(
                        f'<div class="source-card">'
                        f'<div class="source-card-meta"><span>{idx}. {srv} / <code>{tname}</code>{dur_txt}</span>{status_pill_html(ev_status.title(), tone)}</div>'
                        f'<div class="source-card-snippet">{summary}</div>'
                        f"</div>"
                    )
                    for tok in summary_raw.split():
                        tok_clean = tok.strip("\"'()[]:;,.")
                        if tok_clean.lower().endswith((".mp4", ".webm")) and os.path.exists(tok_clean):
                            st.video(tok_clean)
                            break
                        elif tok_clean.lower().endswith((".png", ".jpg", ".jpeg")) and os.path.exists(tok_clean):
                            st.image(tok_clean, caption=f"Generated by {srv}")
                            break


def format_approval_arguments(masked_args_raw: Any) -> tuple[Optional[tuple[str, str, str]], Dict[str, Any]]:
    """Inspect arguments to separate primary code payload from remaining structured parameters."""
    data: Dict[str, Any] = {}
    if isinstance(masked_args_raw, dict):
        data = dict(masked_args_raw)
    elif isinstance(masked_args_raw, str):
        try:
            parsed = json.loads(masked_args_raw)
            if isinstance(parsed, dict):
                data = parsed
            else:
                data = {"value": parsed}
        except Exception:
            return None, {"raw": masked_args_raw}
    else:
        return None, {}

    code_field = None
    for k, v in list(data.items()):
        if isinstance(v, str) and (
            any(sub in k.lower() for sub in ("code", "script", "python", "sql", "command", "bash"))
            or "\n" in v
            or "import " in v
            or "def " in v
            or "class " in v
        ):
            code_field = (k, v)
            break

    if code_field is not None:
        key, code_str = code_field
        rem = {k: v for k, v in data.items() if k != key}
        k_lower = key.lower()
        c_lower = code_str.lower()
        lang = "python" if any(sub in k_lower or sub in c_lower for sub in ("python", "manim", "import ", "def ", "class ")) else "bash"
        return (key, code_str, lang), rem

    return None, data


def render_approval_card(
    pending_approval: Dict[str, Any],
    on_decision: Callable[[str, bool], None],
    key_prefix: str = "chat_appr",
) -> None:
    """Render an interactive Human-in-the-Loop approval prompt with styled code & parameters."""
    approval_id = str(pending_approval.get("approval_id", ""))
    server_name = html.escape(str(pending_approval.get("server_name", "MCP Server")))
    tool_name = html.escape(str(pending_approval.get("tool_name", "tool")))
    category = html.escape(str(pending_approval.get("category", "MODIFICATION")))
    reason = html.escape(str(pending_approval.get("reason", "Action requires user confirmation.")))
    masked_args_raw = pending_approval.get("masked_arguments", "{}")

    cat_badge_cls = "appr-cat-sens" if "SENS" in category.upper() else "appr-cat-mod"

    render_html(
        f'<div class="mcp-appr-card approval-card">'
        f'<div class="mcp-appr-header">'
        f'  <div class="mcp-appr-icon-badge">'
        f'    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">'
        f'      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>'
        f'      <path d="m9 12 2 2 4-4"/>'
        f'    </svg>'
        f'  </div>'
        f'  <div class="mcp-appr-title-group">'
        f'    <div class="mcp-appr-title">Action Requires Authorization</div>'
        f'    <div class="mcp-appr-subtitle">An autonomous AI agent requested to invoke an external MCP tool. Review parameters before approving.</div>'
        f'  </div>'
        f'  <span class="mcp-appr-pill {cat_badge_cls}">{category}</span>'
        f'</div>'
        f'<div class="mcp-appr-meta-strip">'
        f'  <span class="mcp-meta-item">Server: <strong>{server_name}</strong></span>'
        f'  <span class="mcp-meta-divider">&bull;</span>'
        f'  <span class="mcp-meta-item">Tool: <code style="color:#5B61F6;font-weight:600;">{tool_name}</code></span>'
        f'{f"<span class=\"mcp-meta-divider\">&bull;</span><span class=\"mcp-meta-reason\">{reason}</span>" if reason else ""}'
        f'</div>'
        f'</div>'
    )

    code_payload, remaining_params = format_approval_arguments(masked_args_raw)

    if code_payload:
        code_key, code_str, lang = code_payload
        st.caption(f"**Payload Inspection (`{html.escape(code_key)}`) &mdash; Language: {lang.title()}:**")
        st.code(code_str, language=lang, line_numbers=True)
        if remaining_params:
            with st.expander("Additional Tool Arguments", expanded=False):
                st.json(remaining_params)
    else:
        st.caption("**Payload Inspection (JSON):**")
        if remaining_params:
            st.code(json.dumps(remaining_params, indent=2), language="json")
        else:
            st.code(str(masked_args_raw), language="json")

    col_approve, col_reject, col_hint = st.columns([1.5, 1.2, 2.3])
    with col_approve:
        render_html('<div class="appr-btn-wrap">')
        if st.button(
            "✓ Authorize & Execute",
            type="primary",
            icon=":material/check_circle:",
            key=f"{key_prefix}_approve_{approval_id}",
            use_container_width=True,
        ):
            on_decision(approval_id, True)
        render_html("</div>")
    with col_reject:
        render_html('<div class="rej-btn-wrap">')
        if st.button(
            "✕ Deny Action",
            icon=":material/cancel:",
            key=f"{key_prefix}_reject_{approval_id}",
            use_container_width=True,
        ):
            on_decision(approval_id, False)
        render_html("</div>")
    with col_hint:
        render_html(
            '<div style="font-size:12px;color:#64748B;line-height:1.45;padding-top:8px;">'
            'Execution runs strictly with the inspected arguments shown above.'
            '</div>'
        )
