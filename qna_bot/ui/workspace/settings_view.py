"""Workspace settings view covering general, model, API, RAG, security, and data."""

from typing import Callable, Optional

import streamlit as st

from qna_bot.config.settings import AppSettings
from qna_bot.services.document_service import DocumentService
from qna_bot.services.export_service import (
    export_as_json,
    export_as_markdown,
    export_as_text,
)
from qna_bot.services.user_service import UserService
from qna_bot.ui.design_system import (
    render_html,
    render_metric_strip,
    render_page_header,
    render_section_header,
    status_pill_html,
)


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
