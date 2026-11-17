"""Knowledge base document management, uploading, filtering, and deletion view."""

import html
import time
from typing import Any, Dict

import streamlit as st

from qna_bot.config.settings import AppSettings
from qna_bot.services.document_service import DocumentService
from qna_bot.ui.design_system import (
    render_empty_state,
    render_html,
    render_metric_strip,
    render_page_header,
    render_section_header,
    status_pill_html,
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
