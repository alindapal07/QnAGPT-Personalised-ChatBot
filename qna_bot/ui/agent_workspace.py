"""Enterprise Agent Workspace & MCP Execution Monitor view."""

import html
import time
from typing import Callable, Dict, List

import streamlit as st

from qna_bot.services.approval_service import ApprovalService
from qna_bot.services.document_service import DocumentService
from qna_bot.services.mcp_service import MCPService
from qna_bot.ui.components import render_approval_card
from qna_bot.ui.design_system import (
    render_empty_state,
    render_execution_timeline_html,
    render_html,
    render_metric_strip,
    render_page_header,
    render_section_header,
    status_pill_html,
)


def _agent_status_tone(status: str) -> str:
    low = (status or "idle").lower()
    if low in ("completed", "healthy"):
        return "ok"
    if low in ("waiting for approval", "planning", "executing", "processing results", "discovering tools"):
        return "accent"
    if low in ("failed", "error"):
        return "err"
    return "neutral"


def render_agent_workspace_page(
    user_id: str,
    mcp_service: MCPService,
    document_service: DocumentService,
    approval_service: ApprovalService,
    on_approval_decision: Callable[[str, bool], None],
) -> None:
    """Render live Agent status, task overview, timeline, capabilities, approvals, and execution logs."""
    render_page_header(
        "Agent Workspace",
        "Monitor live agent runs, planned execution steps, connected capabilities, pending approvals, and audit logs.",
    )

    mcp_statuses = mcp_service.list_statuses(user_id=user_id)
    mcp_tools = mcp_service.list_tools(user_id=user_id)
    docs = document_service.list_documents(user_id=user_id)
    connected_servers = [s for s in mcp_statuses if s.connected]

    render_metric_strip(
        [
            ("Connected MCP Servers", f"{len(connected_servers)} / {len(mcp_statuses)}"),
            ("Active MCP Tools", len(mcp_tools)),
            ("Built-in Tools", 4),
            ("Indexed Documents", len(docs)),
        ]
    )

    ws = st.session_state.get("agent_workspace_state") or {}
    status = str(ws.get("status") or "Idle")
    task_text = str(ws.get("task") or "").strip()
    planned: List[str] = list(ws.get("planned_steps") or [])
    trace: List[Dict[str, object]] = list(ws.get("execution_trace") or [])

    render_section_header("Active Execution & Status", f"Workspace: {user_id}")
    with st.container(border=True):
        c_task, c_badge = st.columns([4, 1.2])
        with c_task:
            if task_text:
                st.markdown(f"**Latest Task:** {html.escape(task_text)}")
            else:
                st.markdown("**No active agent task yet**")
                st.caption("Submit a prompt in Chat using Auto (Agentic) or Agent Workspace mode to monitor execution here.")
        with c_badge:
            render_html(f'<div style="text-align:right">{status_pill_html(status, _agent_status_tone(status))}</div>')

        if planned:
            st.markdown("**Execution Plan**")
            for idx, step in enumerate(planned, start=1):
                st.markdown(f"{idx}. {step}")

        if trace:
            render_html(render_execution_timeline_html(trace, header="Execution Stages"))

    # ── Capabilities Snapshot ─────────────────────────────────────────────────
    render_section_header("Available Capabilities")
    c_left, c_right = st.columns(2)
    with c_left:
        with st.container(border=True):
            st.markdown("**Built-in Workspace Capabilities**")
            st.caption(
                "- `retrieve_uploaded_documents` & `summarize_uploaded_documents` (FAISS RAG)\n"
                "- `web_search` (Resilient multi-backend web discovery)\n"
                "- `open_web_page` (SSRF-safe HTML & table extraction)"
            )
    with c_right:
        with st.container(border=True):
            st.markdown("**Connected MCP Integrations**")
            if not connected_servers:
                st.caption("No MCP servers currently connected. Visit MCP Marketplace or MCP Servers to connect one.")
            else:
                for srv in connected_servers[:6]:
                    st.caption(f"- **{srv.name}** (`{srv.transport}`) — {srv.tool_count} tools")
                if len(connected_servers) > 6:
                    st.caption(f"+ {len(connected_servers) - 6} more connected servers")

    # ── Pending Approvals ─────────────────────────────────────────────────────
    pending_list = approval_service.list_pending(user_id=user_id)
    active_interrupt = st.session_state.get("pending_approval")
    if active_interrupt or pending_list:
        render_section_header("Pending Human Approvals", "Backend-enforced authorization")
        if active_interrupt:
            render_approval_card(
                active_interrupt,
                on_decision=on_approval_decision,
                key_prefix="ws_appr",
            )
        for rec in pending_list:
            if active_interrupt and rec.approval_id == active_interrupt.get("approval_id"):
                continue
            st.caption(
                f"Pending `{rec.approval_id}` — `{rec.tool_name}` ({rec.category}) | Args: `{rec.masked_arguments}`"
            )

    # ── Execution Logs (Audit History) ────────────────────────────────────────
    render_section_header("Tool Execution History", "Most recent 30 recorded runs")
    audit_rows = mcp_service.list_audit_logs(user_id=user_id, limit=30)
    if not audit_rows:
        render_empty_state(
            "No MCP tool executions recorded",
            "Tool runs from Agent Workspace or Auto mode will appear here with execution duration, masked arguments, and completion status.",
        )
    else:
        f_col1, f_col2 = st.columns([2, 1])
        with f_col1:
            log_query = st.text_input(
                "Filter logs",
                placeholder="Search by tool or server name...",
                label_visibility="collapsed",
                key="ws_log_search",
            ).strip().lower()
        with f_col2:
            status_choice = st.selectbox(
                "Status",
                options=["All", "completed", "failed", "awaiting_approval", "rejected", "blocked"],
                label_visibility="collapsed",
                key="ws_log_status",
            )

        filtered = [
            row
            for row in audit_rows
            if (status_choice == "All" or row.get("status") == status_choice)
            and (
                not log_query
                or log_query in str(row.get("tool_name", "")).lower()
                or log_query in str(row.get("server_name", "")).lower()
            )
        ]
        if not filtered:
            render_empty_state("No matching execution logs", "Adjust the status or search filter above.")
        else:
            for row in filtered:
                started = time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(row.get("started_at") or time.time()),
                )
                dur = row.get("duration_ms")
                dur_str = f"{dur:.0f} ms" if isinstance(dur, (int, float)) else "-"
                st_str = str(row.get("status") or "completed")
                with st.expander(
                    f"{row.get('server_name')} :: {row.get('tool_name')}  |  {st_str}  |  {started}  ({dur_str})",
                    expanded=False,
                ):
                    st.caption("Masked Arguments (Sensitive values redacted)")
                    st.code(str(row.get("masked_arguments") or "{}"), language="json")
                    st.caption("Result Summary")
                    st.write(html.escape(str(row.get("result_summary") or "")))
                    if row.get("error"):
                        st.error(row["error"])

    if ws.get("final_result"):
        render_section_header("Latest Output")
        with st.container(border=True):
            st.markdown(str(ws["final_result"]))
