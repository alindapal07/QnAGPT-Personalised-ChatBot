"""Dedicated Human-in-the-Loop Approval Center workspace page."""

import html
import json
import time
from typing import Any, Callable, Dict, List, Optional

import streamlit as st

from qna_bot.database.models import ApprovalRecord
from qna_bot.services.approval_service import ApprovalService
from qna_bot.services.chat_service import ChatService
from qna_bot.services.session_service import restore_chat
from qna_bot.ui.components import format_approval_arguments
from qna_bot.ui.design_system import (
    render_empty_state,
    render_html,
    render_metric_strip,
    render_page_header,
    render_section_header,
    status_pill_html,
)


def _fmt_ts(ts: Any) -> str:
    if not isinstance(ts, (int, float)) or ts <= 0:
        return "—"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(ts)))


def _status_tone(status: str) -> str:
    status_up = (status or "").upper()
    if status_up == "PENDING":
        return "warn"
    if status_up in ("APPROVED", "CONSUMED"):
        return "ok"
    if status_up == "REJECTED":
        return "err"
    return "neutral"


def render_approval_center_page(
    user_id: str,
    approval_service: ApprovalService,
    chat_service: Optional[ChatService] = None,
    on_decision: Optional[Callable[[str, bool], None]] = None,
) -> None:
    """Render the Human-in-the-Loop Approval Center workspace."""
    render_page_header(
        "Approval Center",
        "Review, authorize, or reject security-sensitive MCP tool execution requests from autonomous agents.",
    )

    all_approvals = approval_service.list_all(user_id=user_id, limit=100)
    pending_cnt = sum(1 for a in all_approvals if a.status == "PENDING")
    approved_cnt = sum(1 for a in all_approvals if a.status == "APPROVED")
    consumed_cnt = sum(1 for a in all_approvals if a.status == "CONSUMED")
    rejected_cnt = sum(1 for a in all_approvals if a.status == "REJECTED")

    render_metric_strip(
        [
            ("Pending Authorization", pending_cnt),
            ("Approved & Consumed", approved_cnt + consumed_cnt),
            ("Rejected Actions", rejected_cnt),
            ("Total Audit Records", len(all_approvals)),
        ]
    )

    col_flt, _ = st.columns([2.5, 4.5])
    with col_flt:
        status_filter = st.selectbox(
            "Filter by Status",
            options=["All Statuses", "Pending Only", "Approved / Consumed", "Rejected Only"],
            key="appr_status_filter",
        )

    filtered = all_approvals
    if status_filter == "Pending Only":
        filtered = [a for a in all_approvals if a.status == "PENDING"]
    elif status_filter == "Approved / Consumed":
        filtered = [a for a in all_approvals if a.status in ("APPROVED", "CONSUMED")]
    elif status_filter == "Rejected Only":
        filtered = [a for a in all_approvals if a.status == "REJECTED"]

    if not filtered:
        render_empty_state(
            title="No approval records found",
            description="When an AI agent requests to invoke an external MCP tool requiring user consent, it will appear here for review.",
        )
        return

    render_section_header("Approval Requests", f"Showing {len(filtered)} records")

    for rec in filtered:
        tone = _status_tone(rec.status)
        pill = status_pill_html(rec.status, tone)
        tool_label = html.escape(f"{rec.server_id}::{rec.tool_name}")
        created_str = _fmt_ts(rec.created_at)

        with st.container(border=True):
            head_col, pill_col = st.columns([5, 1.5])
            with head_col:
                render_html(
                    f'<div style="font-size:14px;font-weight:600;color:#202333;">'
                    f'Tool: <code style="font-size:13px;color:#5B61F6;">{tool_label}</code></div>'
                )
            with pill_col:
                render_html(f'<div style="text-align:right">{pill}</div>')

            meta_col1, meta_col2 = st.columns(2)
            with meta_col1:
                st.caption(f"**Action Category:** {rec.category or 'General'}")
                st.caption(f"**Request Timestamp:** {created_str}")
            with meta_col2:
                st.caption(f"**Thread ID:** `{rec.thread_id}`")
                st.caption(f"**Approval ID:** `{rec.approval_id}`")

            if rec.reason:
                st.markdown(f"**Reason for Execution:** {rec.reason}")

            if rec.masked_arguments:
                code_payload, other_params = format_approval_arguments(rec.masked_arguments)
                if code_payload:
                    ckey, cval, clang = code_payload
                    with st.expander(f"Code Payload ({ckey}) &mdash; {clang.title()}", expanded=True):
                        st.code(cval, language=clang, line_numbers=True)
                    if other_params:
                        with st.expander("Additional Parameters", expanded=False):
                            st.json(other_params)
                else:
                    with st.expander("Inspected Arguments (Sensitive Values Masked)", expanded=False):
                        if other_params:
                            st.code(json.dumps(other_params, indent=2), language="json")
                        else:
                            st.code(rec.masked_arguments, language="json")

            if rec.status == "PENDING":
                c_appr, c_rej, c_jump = st.columns([1.5, 1.5, 3])
                with c_appr:
                    render_html('<div class="appr-btn-wrap">')
                    if st.button(
                        "✓ Approve Execution",
                        key=f"ac_btn_appr_{rec.approval_id}",
                        type="primary",
                        icon=":material/check_circle:",
                        use_container_width=True,
                    ):
                        if on_decision is not None:
                            on_decision(rec.approval_id, True)
                        else:
                            approval_service.record_decision(rec.approval_id, user_id=user_id, approved=True)
                        st.toast("Action approved.")
                        st.rerun()
                    render_html("</div>")

                with c_rej:
                    render_html('<div class="rej-btn-wrap">')
                    if st.button(
                        "✕ Reject Action",
                        key=f"ac_btn_rej_{rec.approval_id}",
                        type="secondary",
                        icon=":material/cancel:",
                        use_container_width=True,
                    ):
                        if on_decision is not None:
                            on_decision(rec.approval_id, False)
                        else:
                            approval_service.record_decision(rec.approval_id, user_id=user_id, approved=False)
                        st.toast("Action rejected.")
                        st.rerun()
                    render_html("</div>")

                with c_jump:
                    if chat_service is not None and st.button(
                        "Open Conversation",
                        key=f"ac_jump_{rec.approval_id}",
                        type="tertiary",
                        icon=":material/chat:",
                    ):
                        msgs = chat_service.load_thread_history(rec.thread_id, user_id=user_id)
                        restore_chat({"id": rec.thread_id, "messages": msgs})
                        st.session_state.active_page = "Chat"
                        st.rerun()
