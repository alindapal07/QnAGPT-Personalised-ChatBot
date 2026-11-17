"""Tool Marketplace view for tool discovery, policy governance, and inspection."""

import html
from typing import Dict

import streamlit as st

from qna_bot.services.mcp_service import MCPService
from qna_bot.ui.design_system import (
    render_empty_state,
    render_html,
    render_metric_strip,
    render_page_header,
    render_section_header,
    status_pill_html,
)

BUILTIN_TOOL_SPECS = (
    ("retrieve_uploaded_documents", "READ_ONLY", "Semantic vector search across your user-scoped Qdrant Knowledge Base."),
    ("summarize_uploaded_documents", "READ_ONLY", "Workspace document inventory and representative chunk excerpts."),
    ("web_search", "READ_ONLY", "Multi-backend public web discovery returning numbered source URLs."),
    ("open_web_page", "READ_ONLY", "SSRF-protected HTTPS webpage fetcher extracting clean text and Markdown tables."),
)


def render_tool_marketplace_page(user_id: str, mcp_service: MCPService) -> None:
    """Render Tool Discovery & Policy Governance separate from MCP Server lifecycle."""
    render_page_header(
        "Tool Marketplace",
        "Inspect tool schemas, toggle availability, configure execution policies, and review per-tool audit activity.",
    )

    all_tools = mcp_service.list_tools(user_id=user_id, include_disabled=True)
    enabled_cnt = sum(1 for t in all_tools if t.enabled and t.permission_category != "BLOCKED")
    blocked_cnt = sum(1 for t in all_tools if t.permission_category == "BLOCKED")
    render_metric_strip(
        [
            ("Built-in Tools", len(BUILTIN_TOOL_SPECS)),
            ("Connected MCP Tools", len(all_tools)),
            ("Enabled MCP Tools", enabled_cnt),
            ("Blocked by Policy", blocked_cnt),
        ]
    )

    render_section_header("Built-in Workspace Tools", "Always enabled for QnAGPT agents")
    b_cols = st.columns(2)
    for idx, (t_name, t_cat, t_desc) in enumerate(BUILTIN_TOOL_SPECS):
        with b_cols[idx % 2]:
            with st.container(border=True):
                render_html(
                    f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">'
                    f'<code>{html.escape(t_name)}</code> {status_pill_html(t_cat, "ok")}'
                    f"</div>"
                )
                st.caption(t_desc)

    render_section_header("Connected MCP Tools", f"{len(all_tools)} discovered")
    if not all_tools:
        render_empty_state(
            "No external MCP tools discovered",
            "Connect an MCP server from MCP Marketplace or MCP Servers to inspect its schemas and governance policies here.",
        )
        return

    server_names = sorted({t.server_name for t in all_tools})
    c_search, c_srv, c_cat, c_avail = st.columns([2.2, 1.3, 1.3, 1.2])
    with c_search:
        query = st.text_input(
            "Search tools",
            placeholder="Filter by tool name or description...",
            label_visibility="collapsed",
            key="mkt_filter",
        ).strip().lower()
    with c_srv:
        srv_filter = st.selectbox("Server", options=["All", *server_names], key="tmkt_srv_filter")
    with c_cat:
        cat_filter = st.selectbox(
            "Category",
            options=["All", "READ_ONLY", "MODIFICATION", "SENSITIVE", "BLOCKED"],
            key="tmkt_cat_filter",
        )
    with c_avail:
        avail_filter = st.selectbox("State", options=["All", "Enabled", "Disabled"], key="tmkt_avail_filter")

    tools = [
        t
        for t in all_tools
        if (not query or query in t.original_name.lower() or query in t.server_name.lower() or query in t.description.lower())
        and (srv_filter == "All" or t.server_name == srv_filter)
        and (cat_filter == "All" or t.permission_category == cat_filter)
        and (avail_filter == "All" or (avail_filter == "Enabled" and t.enabled) or (avail_filter == "Disabled" and not t.enabled))
    ]

    if not tools:
        render_empty_state("No tools match the active filters", "Broaden your search or reset the category/server dropdowns.")
        return

    audit_rows = mcp_service.list_audit_logs(user_id=user_id, limit=50)
    exec_counts: Dict[str, int] = {}
    for row in audit_rows:
        key = f"{row.get('server_name')}::{row.get('tool_name')}"
        exec_counts[key] = exec_counts.get(key, 0) + 1

    if len(tools) > 50 and not query:
        st.caption(f"Showing first 50 of {len(tools)} matching tools. Search above to narrow down.")
        tools = tools[:50]

    categories = ("READ_ONLY", "MODIFICATION", "SENSITIVE", "BLOCKED")
    for tool in tools:
        cat_tone = "ok" if tool.permission_category == "READ_ONLY" else (
            "err" if tool.permission_category == "BLOCKED" else "warn"
        )
        en_tone = "ok" if tool.enabled else "neutral"
        en_label = "Enabled" if tool.enabled else "Disabled"
        runs = exec_counts.get(f"{tool.server_name}::{tool.original_name}", 0)

        with st.container(border=True):
            render_html(
                f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:2px;">'
                f'<strong style="font-size:13.5px;color:#172033;">{html.escape(tool.original_name)}</strong> '
                f"{status_pill_html(tool.server_name, 'accent')} "
                f"{status_pill_html(tool.permission_category, cat_tone)} "
                f"{status_pill_html(en_label, en_tone)}"
                f"</div>"
            )
            st.caption(f"{tool.description or 'No description provided.'}  |  Recent runs: {runs}")
            c_en, c_cat_box, c_auto, c_save = st.columns([0.9, 1.8, 1.5, 0.9])
            with c_en:
                new_en = st.checkbox("Enabled", value=tool.enabled, key=f"perm_en_{tool.namespaced_name}")
            with c_cat_box:
                new_cat = st.selectbox(
                    "Permission Category",
                    options=list(categories),
                    index=categories.index(tool.permission_category),
                    key=f"perm_cat_{tool.namespaced_name}",
                )
            with c_auto:
                new_auto = st.checkbox(
                    "Auto-approve (READ_ONLY/MOD only)",
                    value=tool.auto_approve,
                    disabled=(new_cat in ("SENSITIVE", "BLOCKED")),
                    key=f"perm_auto_{tool.namespaced_name}",
                )
            with c_save:
                if st.button("Apply", type="primary", key=f"perm_save_{tool.namespaced_name}", use_container_width=True):
                    mcp_service.set_tool_permission(
                        user_id=user_id,
                        server_id=tool.server_id,
                        tool_name=tool.original_name,
                        category=new_cat,  # type: ignore[arg-type]
                        auto_approve=(new_auto and new_cat not in ("SENSITIVE", "BLOCKED")),
                        enabled=new_en,
                    )
                    st.toast("Tool policy updated.")
                    st.rerun()

            with st.expander("Schema & Details", expanded=False):
                st.caption(f"LangGraph Tool Identifier: `{tool.namespaced_name}` | Auto-approve: `{tool.auto_approve}`")
                st.json(tool.input_schema or {})
