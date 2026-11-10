"""Enterprise Workspace views: Knowledge Base, Tool Marketplace, and Settings."""

import html
import re
import time
from typing import Any, Callable, Dict, List, Optional
import uuid

import streamlit as st

from qna_bot.config.settings import AppSettings
from qna_bot.services.document_service import DocumentService
from qna_bot.services.export_service import (
    export_as_json,
    export_as_markdown,
    export_as_text,
)
from qna_bot.services.mcp_service import MCPService
from qna_bot.services.user_service import UserService
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


def _fmt_size(size_bytes: Any) -> str:
    if not isinstance(size_bytes, (int, float)) or size_bytes <= 0:
        return "—"
    if size_bytes < 1024:
        return f"{int(size_bytes)} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def _fmt_ts(ts: Any) -> str:
    if not isinstance(ts, (int, float)) or ts <= 0:
        return "—"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(ts)))


def render_knowledge_base_page(
    settings: AppSettings,
    user_id: str,
    document_service: DocumentService,
) -> None:
    """Render the dedicated Knowledge Base upload, filtering, inspection, reprocessing, and delete UI."""
    render_page_header(
        "Knowledge Base",
        f"Index PDF, TXT, DOCX, CSV, or MD documents into Qdrant ({settings.embedding_model_name}) with local original storage.",
    )

    vec_health = document_service.vector_health()
    if not vec_health.get("ok", False):
        st.warning(
            f"Qdrant ({vec_health.get('mode', settings.qdrant_mode).upper()} @ `{vec_health.get('url', settings.qdrant_url)}`) "
            f"is currently unreachable: {vec_health.get('detail', 'Connection refused')}. "
            "Original files uploaded below will still be safely saved locally so you can click Retry Indexing once Qdrant is ready."
        )

    docs = document_service.list_documents(user_id=user_id)
    total_chunks = sum(int(d.get("chunk_count") or 0) for d in docs)
    total_pages = sum(int(d.get("page_count") or 0) for d in docs)
    qdrant_badge = f"Qdrant ({str(vec_health.get('mode', settings.qdrant_mode)).title()})"
    render_metric_strip(
        [
            ("Uploaded Documents", len(docs)),
            ("Total Vector Chunks", total_chunks),
            ("Total Pages", total_pages or "—"),
            ("Vector Store", f"{qdrant_badge} · {'Online' if vec_health.get('ok') else 'Offline'}"),
        ]
    )

    render_section_header("Upload & Index Documents", f"Max size: {settings.max_document_size_mb} MB per file")
    nonce = st.session_state.get("doc_uploader_nonce", 0)
    uploaded_docs = st.file_uploader(
        "Drag and drop files here or browse",
        type=list(settings.allowed_document_types),
        accept_multiple_files=True,
        key=f"kb_file_uploader_{nonce}",
    )

    if uploaded_docs:
        staged_names = ", ".join(f.name for f in uploaded_docs)
        render_html(
            f'<div style="margin:4px 0 8px">{status_pill_html("Uploaded", "accent")} '
            f'<span style="font-size:12.5px;color:#64748B;">Ready to process: <code>{html.escape(staged_names)}</code></span></div>'
        )
        if st.button("Process & Index Documents", type="primary", icon=":material/upload_file:", key="kb_btn_ingest"):
            payload = [(f.name, f.getvalue()) for f in uploaded_docs]
            reprocess_cache: Dict[str, bytes] = st.session_state.setdefault("kb_session_file_cache", {})
            for name, raw in payload:
                reprocess_cache[name] = raw

            with st.spinner("Processing: storing locally, extracting text, and indexing into Qdrant..."):
                results = document_service.ingest_files(payload, user_id=user_id)

            st.session_state.doc_uploader_nonce = nonce + 1
            st.session_state["kb_ingest_feedback"] = [
                (
                    r.status,
                    r.filename,
                    f"Stored & indexed ({r.chunk_count} chunks)"
                    if r.status == "ingested"
                    else (
                        "Already indexed (duplicate hash skipped)"
                        if r.status == "duplicate"
                        else r.error
                    ),
                )
                for r in results
            ]
            st.rerun()

    for r_status, fname, detail in st.session_state.pop("kb_ingest_feedback", []):
        if r_status == "ingested":
            st.success(f"{fname}: {detail}")
        elif r_status == "duplicate":
            st.info(f"{fname}: {detail}")
        else:
            st.error(f"{fname}: {detail}")

    render_section_header("Stored & Indexed Document Repository", f"{len(docs)} documents")
    if not docs:
        render_empty_state(
            "Your Knowledge Base is empty",
            "Upload PDF, TXT, DOCX, CSV, or MD files above to store originals locally and enable Qdrant semantic retrieval.",
        )
        return

    c_search, c_type = st.columns([3, 1.2])
    with c_search:
        kb_query = st.text_input(
            "Search documents",
            placeholder="Filter by filename...",
            label_visibility="collapsed",
            key="kb_search_input",
        ).strip().lower()
    with c_type:
        type_filter = st.selectbox(
            "File Type",
            options=["All", "pdf", "txt", "docx", "csv", "md"],
            format_func=lambda x: "All Types" if x == "All" else x.upper(),
            label_visibility="collapsed",
            key="kb_type_filter",
        )

    filtered_docs = [
        d
        for d in docs
        if (not kb_query or kb_query in str(d.get("filename") or d.get("source") or "").lower())
        and (
            type_filter == "All"
            or str(d.get("file_type", "")).lower().lstrip(".") == type_filter
            or str(d.get("filename") or d.get("source") or "").lower().endswith(f".{type_filter}")
        )
    ]

    if not filtered_docs:
        render_empty_state("No matching documents", "Try clearing your filename search or file-type filter.")
        return

    session_cache: Dict[str, bytes] = st.session_state.get("kb_session_file_cache") or {}
    for doc in filtered_docs:
        doc_id = str(doc.get("document_id", ""))
        fname = str(doc.get("filename") or doc.get("source") or "Document")
        ftype = str(doc.get("file_type") or (fname.rsplit(".", 1)[-1] if "." in fname else "file")).upper().lstrip(".")
        size_str = _fmt_size(doc.get("file_size_bytes"))
        uploaded_str = _fmt_ts(doc.get("created_at"))
        chunks = int(doc.get("chunk_count") or 0)
        pages = doc.get("page_count")
        page_str = f"{pages} pages" if pages else "—"
        proc_status = str(doc.get("processing_status") or "indexed").lower()
        pill_variant = "ok" if proc_status == "indexed" else ("err" if proc_status == "failed" else "warn")

        with st.container(border=True):
            top_info, top_act = st.columns([3.8, 2.2])
            with top_info:
                render_html(
                    f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:2px;">'
                    f'<strong style="font-size:13.5px;color:#172033;">{html.escape(fname)}</strong> '
                    f"{status_pill_html(ftype, 'neutral')} {status_pill_html(proc_status.title(), pill_variant)}"
                    f"</div>"
                )
                st.caption(
                    f"Size: {size_str}  |  Uploaded: {uploaded_str}  |  Pages: {page_str}  |  Chunks: {chunks}"
                )
                if proc_status == "failed" and doc.get("error_message"):
                    st.caption(f"Indexing error: {doc['error_message']}")
            with top_act:
                b_re, b_dl, b_del = st.columns(3)
                with b_re:
                    if doc.get("storage_path"):
                        if st.button(
                            "Retry" if proc_status == "failed" else "Re-index",
                            icon=":material/refresh:",
                            key=f"kb_retry_{doc_id}",
                            use_container_width=True,
                        ):
                            res = document_service.retry_document(doc_id, user_id=user_id)
                            detail_msg = (
                                f"Re-indexed ({res.chunk_count} chunks)"
                                if res.status == "ingested"
                                else res.error
                            )
                            st.session_state["kb_ingest_feedback"] = [
                                (res.status, res.filename, detail_msg)
                            ]
                            st.rerun()
                    elif fname in session_cache and st.button(
                        "Reprocess",
                        icon=":material/refresh:",
                        key=f"kb_reproc_{doc_id}",
                        use_container_width=True,
                    ):
                        raw = session_cache[fname]
                        document_service.remove_document(doc_id, user_id=user_id)
                        res_list = document_service.ingest_files([(fname, raw)], user_id=user_id)
                        st.session_state["kb_ingest_feedback"] = [
                            (r.status, r.filename, f"Reprocessed ({r.chunk_count} chunks)" if r.status == "ingested" else r.error)
                            for r in res_list
                        ]
                        st.rerun()
                with b_dl:
                    orig = None
                    try:
                        orig = document_service.read_original_file(doc_id, user_id=user_id)
                    except Exception:
                        orig = None
                    if orig is not None:
                        st.download_button(
                            "File",
                            data=orig[1],
                            file_name=orig[0],
                            key=f"kb_dl_{doc_id}",
                            icon=":material/download:",
                            use_container_width=True,
                        )
                with b_del:
                    confirm_key = st.session_state.get("kb_confirm_del_id")
                    if confirm_key != doc_id:
                        if st.button(
                            "Delete",
                            icon=":material/delete:",
                            key=f"kb_del_ask_{doc_id}",
                            use_container_width=True,
                        ):
                            st.session_state["kb_confirm_del_id"] = doc_id
                            st.rerun()
                    else:
                        if st.button(
                            "Confirm",
                            type="primary",
                            key=f"kb_del_yes_{doc_id}",
                            use_container_width=True,
                        ):
                            document_service.remove_document(doc_id, user_id=user_id)
                            st.session_state["kb_confirm_del_id"] = None
                            st.toast(f"Removed {fname}")
                            st.rerun()

            with st.expander("Document Details", expanded=False):
                st.caption(
                    f"Document ID: `{doc_id}` | SHA-256 Prefix: `{str(doc.get('file_hash', ''))[:16]}` | "
                    f"User Namespace: `{user_id}`"
                )

    st.divider()
    if not st.session_state.get("kb_confirm_clear"):
        if st.button("Clear All Documents", icon=":material/delete:", key="kb_clear_btn"):
            st.session_state["kb_confirm_clear"] = True
            st.rerun()
    else:
        st.warning("Permanently remove all indexed documents from this workspace?")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Confirm Clear Index", type="primary", key="kb_clear_yes"):
                document_service.clear_index(user_id=user_id)
                st.session_state["kb_confirm_clear"] = False
                st.rerun()
        with c2:
            if st.button("Cancel", key="kb_clear_no"):
                st.session_state["kb_confirm_clear"] = False
                st.rerun()


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


def render_settings_page(
    settings: AppSettings,
    user_id: str,
    on_clear_chat: Callable[[], None],
    user_service: Optional[UserService] = None,
    document_service: Optional[DocumentService] = None,
) -> None:
    """Render structured SaaS Settings covering General, Appearance, Model, API, MCP, RAG, Security & Data."""
    render_page_header(
        "Settings",
        "Manage authenticated account preferences, Groq models, API readiness, Qdrant, and data export.",
    )

    tab_gen, tab_model, tab_api_mcp, tab_rag_sec, tab_data = st.tabs(
        ["General & Appearance", "Model Configuration", "API & MCP", "RAG & Security", "Data Management"]
    )

    with tab_gen:
        render_section_header("Authenticated Workspace Identity")
        with st.container(border=True):
            st.write(f"- **User Namespace ID:** `{user_id}`")
            st.caption("Workspace ownership is bound strictly to your authenticated session and enforced in SQLite, local file storage, and Qdrant filters.")
            if st.button("Open Account & Security", icon=":material/person:", key="settings_open_acct"):
                st.session_state.active_page = "Account"
                st.rerun()

        render_section_header("Appearance")
        with st.container(border=True):
            render_html(
                f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">'
                f'<strong style="font-size:13.5px;color:#172033;">Theme Mode:</strong> '
                f"{status_pill_html('Light Workspace (Active)', 'accent')}"
                f"</div>"
            )
            st.caption("QnAGPT enforces a consistent high-contrast light interface designed for extended reading.")

    with tab_model:
        render_section_header("Groq LLM Configuration")
        current_model = str(st.session_state.get("cfg_text_model") or settings.default_text_model)
        idx = (
            list(settings.available_text_models).index(current_model)
            if current_model in settings.available_text_models
            else 0
        )
        sel_model = st.selectbox(
            "Default Text & Tool Model",
            options=list(settings.available_text_models),
            index=idx,
            key="set_model_select",
            help="Used by both the Agent LLM (tool_choice='auto') and Response LLM (final synthesis).",
        )
        st.caption(f"Multimodal Vision Model (used when images are attached): `{settings.default_vision_model}`")

        c_temp, c_tok = st.columns(2)
        with c_temp:
            sel_temp = st.slider(
                "Temperature",
                0.0,
                2.0,
                float(st.session_state.get("cfg_temperature", settings.default_temperature)),
                step=0.05,
                key="set_temp_slider",
            )
        with c_tok:
            sel_tok = st.slider(
                "Max Output Tokens",
                256,
                4096,
                int(st.session_state.get("cfg_max_tokens", settings.default_max_tokens)),
                step=256,
                key="set_tok_slider",
            )
        sel_sys = st.text_area(
            "Custom Workspace Instructions",
            value=str(st.session_state.get("cfg_system_prompt", settings.default_system_prompt)),
            height=115,
            key="set_sys_prompt",
        )
        st.session_state["cfg_text_model"] = sel_model
        st.session_state["cfg_temperature"] = sel_temp
        st.session_state["cfg_max_tokens"] = sel_tok
        st.session_state["cfg_system_prompt"] = sel_sys

        bm1, bm2, _ = st.columns([1.2, 1.2, 2.6])
        with bm1:
            if st.button("Save Model Settings", type="primary", key="set_save_model", use_container_width=True):
                if user_service is not None:
                    user_service.save_settings(
                        user_id,
                        {
                            "text_model": sel_model,
                            "temperature": str(sel_temp),
                            "max_tokens": str(sel_tok),
                            "system_prompt": sel_sys,
                        },
                    )
                st.toast(f"Model configuration saved ({sel_model}).")
        with bm2:
            if st.button("Reset Defaults", key="set_reset_model", use_container_width=True):
                st.session_state["cfg_text_model"] = settings.default_text_model
                st.session_state["cfg_temperature"] = settings.default_temperature
                st.session_state["cfg_max_tokens"] = settings.default_max_tokens
                st.session_state["cfg_system_prompt"] = settings.default_system_prompt
                if user_service is not None:
                    user_service.save_settings(
                        user_id,
                        {
                            "text_model": settings.default_text_model,
                            "temperature": str(settings.default_temperature),
                            "max_tokens": str(settings.default_max_tokens),
                            "system_prompt": settings.default_system_prompt,
                        },
                    )
                st.toast("Model settings reset to workspace defaults.")
                st.rerun()

    with tab_api_mcp:
        render_section_header("API & OAuth Credentials Status", "Secrets are never exposed in the UI")
        with st.container(border=True):
            groq_ok = bool((settings.groq_api_key or "").strip())
            glama_ok = bool((settings.glama_api_key or "").strip())
            google_ok = bool(settings.google_client_id.strip() and settings.google_client_secret.strip())
            github_ok = bool(settings.github_client_id.strip() and settings.github_client_secret.strip())
            qdrant_key_ok = bool((settings.qdrant_api_key or "").strip())

            groq_pill = status_pill_html("Configured in .env / env", "ok") if groq_ok else status_pill_html("Not Set", "err")
            glama_pill = status_pill_html("Configured in .env / env", "ok") if glama_ok else status_pill_html("Optional / Not Set", "warn")
            google_pill = status_pill_html("Configured", "ok") if google_ok else status_pill_html("Not Set", "neutral")
            github_pill = status_pill_html("Configured", "ok") if github_ok else status_pill_html("Not Set", "neutral")
            qkey_pill = status_pill_html("Configured", "ok") if qdrant_key_ok else status_pill_html("Local / No Key", "neutral")

            render_html(
                f'<div style="display:flex;flex-direction:column;gap:6px;font-size:13px;color:#172033;">'
                f'<div><strong>Groq API Key (<code>GROQ_API_KEY</code>):</strong> {groq_pill}</div>'
                f'<div><strong>Glama Directory Key (<code>GLAMA_API_KEY</code>):</strong> {glama_pill}</div>'
                f'<div><strong>Google OAuth (<code>GOOGLE_CLIENT_ID</code>):</strong> {google_pill}</div>'
                f'<div><strong>GitHub OAuth (<code>GITHUB_CLIENT_ID</code>):</strong> {github_pill}</div>'
                f'<div><strong>Qdrant API Key (<code>QDRANT_API_KEY</code>):</strong> {qkey_pill}</div>'
                f"</div>"
            )
            st.caption("Configure keys in your local `.env` file and save; QnAGPT hot-reloads `.env` automatically without exposing raw values.")

        render_section_header("Agent & MCP Runtime Limits")
        st.session_state["cfg_max_iterations"] = st.slider(
            "Maximum Agent Iterations per Turn",
            min_value=1,
            max_value=20,
            value=int(st.session_state.get("cfg_max_iterations", settings.max_agent_iterations)),
        )
        render_metric_strip(
            [
                ("MCP Connect Timeout", f"{settings.mcp_connect_timeout_sec}s"),
                ("MCP Tool Timeout", f"{settings.mcp_tool_timeout_sec}s"),
                ("Max Tool Output", f"{settings.mcp_max_output_chars} chars"),
            ]
        )

    with tab_rag_sec:
        render_section_header("RAG & Qdrant Vector Store Configuration")
        with st.container(border=True):
            vh = document_service.vector_health() if document_service is not None else {"ok": False, "mode": settings.qdrant_mode, "url": settings.qdrant_url}
            st.write(f"- Vector Mode: `{settings.qdrant_mode}` | Endpoint: `{settings.qdrant_url}` | Status: `{'Connected' if vh.get('ok') else 'Offline'}`")
            st.write(f"- Collection: `{settings.qdrant_collection}` | Embedding Model: `{settings.embedding_model_name}`")
            st.write(f"- Chunking Strategy: `{settings.chunk_size}` chars | `{settings.chunk_overlap}` char overlap")
            st.write(f"- Retrieval Top-K: `{settings.retrieval_top_k}` | Score Threshold: `{settings.retrieval_score_threshold}`")
            st.write(f"- Local Storage Root: `{settings.storage_dir / 'users'}`")

        render_section_header("Security & Isolation Guardrails")
        with st.container(border=True):
            st.write("- **Human-in-the-Loop:** Mandatory backend approval for `MODIFICATION` & `SENSITIVE` MCP actions.")
            st.write("- **Credential Storage:** Fernet AES-128-CBC + HMAC-SHA256 encrypted vault per `(user_id, server_id)`.")
            st.write("- **Web & Remote SSRF Guard:** HTTPS enforcement, DNS/private-IP blocking, and redirect validation on webpage fetches.")

    with tab_data:
        render_section_header("SQLite Persistence & Conversation Export")
        st.caption(f"SQLite Storage Path: `{settings.sqlite_db_path}`")
        messages = st.session_state.get("messages") or []
        if messages:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.download_button(
                    "Export JSON",
                    data=export_as_json(st.session_state.current_chat_id, messages),
                    file_name="qnagpt_chat.json",
                    mime="application/json",
                    icon=":material/download:",
                    use_container_width=True,
                )
            with c2:
                st.download_button(
                    "Export Markdown",
                    data=export_as_markdown(messages),
                    file_name="qnagpt_chat.md",
                    mime="text/markdown",
                    icon=":material/download:",
                    use_container_width=True,
                )
            with c3:
                st.download_button(
                    "Export TXT",
                    data=export_as_text(messages),
                    file_name="qnagpt_chat.txt",
                    mime="text/plain",
                    icon=":material/download:",
                    use_container_width=True,
                )
        else:
            st.info("Active conversation has no messages to export yet.")

        st.divider()
        if not st.session_state.get("confirm_clear"):
            if st.button("Clear Current Conversation", icon=":material/delete:", key="settings_clear_conv"):
                st.session_state.confirm_clear = True
                st.rerun()
        else:
            st.warning("Clear all messages in the active conversation?")
            cy, cn = st.columns(2)
            with cy:
                if st.button("Confirm Clear", type="primary", key="settings_clear_yes"):
                    on_clear_chat()
            with cn:
                if st.button("Cancel", key="settings_clear_no"):
                    st.session_state.confirm_clear = False
                    st.rerun()
