"""Application entry point and Streamlit workspace orchestration."""

from pathlib import Path
import base64
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Ensure repository root is importable when launched via `streamlit run qna_bot/app.py`,
# and strip `qna_bot/` itself from sys.path so `qna_bot/mcp` cannot shadow the PyPI `mcp` SDK.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from qna_bot import sanitize_import_path  # noqa: E402

sanitize_import_path()

import streamlit as st  # noqa: E402

from qna_bot.agent.approval import ApprovalManager  # noqa: E402
from qna_bot.agent.memory import ConversationMemoryManager  # noqa: E402
from qna_bot.config.settings import AppSettings, get_settings  # noqa: E402
from qna_bot.core.logging_config import configure_logging, get_logger  # noqa: E402
from qna_bot.database.connection import DatabaseManager  # noqa: E402
from qna_bot.database.legacy_migration import ensure_legacy_workspace_synced  # noqa: E402
from qna_bot.database.session import SQLAlchemyDatabase  # noqa: E402
from qna_bot.mcp.client_manager import MCPClientManager  # noqa: E402
from qna_bot.mcp.registry.service import MCPRegistryService  # noqa: E402
from qna_bot.rag.qdrant_store import QdrantVectorStoreManager  # noqa: E402
from qna_bot.services.agent_service import AIAgent  # noqa: E402
from qna_bot.services.approval_service import ApprovalService  # noqa: E402
from qna_bot.services.auth_service import AuthService  # noqa: E402
from qna_bot.services.chat_service import ChatService, ChatTurnResult  # noqa: E402
from qna_bot.services.conversation_service import ConversationService  # noqa: E402
from qna_bot.services.document_service import DocumentService  # noqa: E402
from qna_bot.services.mcp_service import MCPService  # noqa: E402
from qna_bot.services.session_service import (  # noqa: E402
    _derive_chat_title,
    clear_current_chat,
    initialize_session_state,
    start_new_chat,
)
from qna_bot.services.suggestion_service import SuggestionService  # noqa: E402
from qna_bot.services.user_service import UserService  # noqa: E402
from qna_bot.storage.local import LocalStorageProvider  # noqa: E402
from qna_bot.ui import (  # noqa: E402
    SidebarConfig,
    apply_styles,
    handle_oauth_callback_params,
    render_account_page,
    render_agent_workspace_page,
    render_approval_card,
    render_auth_gate,
    render_chat_composer,
    render_chat_history,
    render_execution_timeline_html,
    render_header,
    render_knowledge_base_page,
    render_mcp_dashboard,
    render_mcp_marketplace_page,
    render_settings_page,
    render_sidebar,
    render_sources_panel,
    render_tool_marketplace_page,
    render_user_message,
    render_welcome_screen,
)
from qna_bot.ui.design_system import render_html  # noqa: E402

logger = get_logger(__name__)


@st.cache_resource
def _get_shared_orm_db(db_url: str) -> SQLAlchemyDatabase:
    orm_db = SQLAlchemyDatabase(db_url_or_path=db_url, auto_create=True)
    ensure_legacy_workspace_synced(orm_db)
    return orm_db


@st.cache_resource
def _get_shared_storage(storage_dir: str) -> LocalStorageProvider:
    return LocalStorageProvider(root_dir=Path(storage_dir))


@st.cache_resource
def _get_shared_db(db_path: str) -> DatabaseManager:
    return DatabaseManager(db_path=Path(db_path))


@st.cache_resource
def _get_shared_memory(db_path: str) -> ConversationMemoryManager:
    return ConversationMemoryManager(db_path=Path(db_path))


@st.cache_resource
def _get_shared_vector_store(
    qdrant_mode: str,
    qdrant_url: str,
    qdrant_api_key: str,
    qdrant_collection: str,
    embedding_model: str,
) -> QdrantVectorStoreManager:
    cfg = get_settings()
    custom_cfg = AppSettings(
        groq_api_key=cfg.groq_api_key,
        embedding_model_name=embedding_model,
        qdrant_mode=qdrant_mode,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        qdrant_collection=qdrant_collection,
    )
    return QdrantVectorStoreManager(settings=custom_cfg)


@st.cache_resource
def _get_shared_mcp_manager(
    db_path: str,
    connect_timeout: int,
    tool_timeout: int,
) -> MCPClientManager:
    db = _get_shared_db(db_path)
    return MCPClientManager(
        db_manager=db,
        connect_timeout=connect_timeout,
        tool_timeout=tool_timeout,
    )


@st.cache_resource
def _get_shared_registry_service(
    db_path: str,
    official_url: str,
    glama_url: str,
    glama_key: str,
    cache_ttl: int,
) -> MCPRegistryService:
    db = _get_shared_db(db_path)
    cfg = get_settings()
    # Respect custom test/runtime overrides cleanly
    custom_cfg = AppSettings(
        groq_api_key=cfg.groq_api_key,
        data_dir=Path(db_path).parent,
        vector_store_dir=cfg.vector_store_dir,
        sqlite_db_path=Path(db_path),
        official_registry_url=official_url,
        glama_api_url=glama_url,
        glama_api_key=glama_key,
        registry_cache_ttl_sec=cache_ttl,
    )
    return MCPRegistryService(db_manager=db, settings=custom_cfg)


def build_platform_services(
    settings: AppSettings,
) -> tuple[SQLAlchemyDatabase, LocalStorageProvider, QdrantVectorStoreManager, AuthService, UserService]:
    """Instantiate authenticated platform foundation services backed by SQLAlchemy + Qdrant + LocalStorage."""
    orm_db = _get_shared_orm_db(settings.resolved_database_url)
    storage = _get_shared_storage(str(settings.storage_dir))
    vector_store = _get_shared_vector_store(
        settings.qdrant_mode,
        settings.qdrant_url,
        settings.qdrant_api_key,
        settings.qdrant_collection,
        settings.embedding_model_name,
    )
    auth_svc = AuthService(db=orm_db, settings=settings)
    user_svc = UserService(db=orm_db, storage=storage, vector_store=vector_store)
    return orm_db, storage, vector_store, auth_svc, user_svc


def build_runtime_services(
    settings: AppSettings,
) -> tuple[ChatService, DocumentService, MCPService, ApprovalService, ConversationService]:
    """Instantiate backend workspace services using process-wide cached managers."""
    db_path_str = str(settings.sqlite_db_path)
    orm_db, storage, vector_store, _, _ = build_platform_services(settings)
    db = _get_shared_db(db_path_str)
    memory = _get_shared_memory(str(settings.langgraph_checkpoint_db))
    mcp_manager = _get_shared_mcp_manager(
        db_path_str,
        settings.mcp_connect_timeout_sec,
        settings.mcp_tool_timeout_sec,
    )
    registry_svc = _get_shared_registry_service(
        db_path_str,
        settings.official_registry_url,
        settings.glama_api_url,
        settings.glama_api_key,
        settings.registry_cache_ttl_sec,
    )
    approval_mgr = ApprovalManager(db_manager=db)

    chat_svc = ChatService(
        settings=settings,
        memory_manager=memory,
        vector_store=vector_store,
        mcp_manager=mcp_manager,
        approval_manager=approval_mgr,
        db=orm_db,
    )
    doc_svc = DocumentService(
        settings=settings,
        vector_store=vector_store,
        storage=storage,
        db=orm_db,
    )
    mcp_svc = MCPService(
        client_manager=mcp_manager,
        db_manager=db,
        registry_service=registry_svc,
        settings=settings,
    )
    appr_svc = ApprovalService(approval_manager=approval_mgr, db_manager=db)
    conv_svc = ConversationService(memory_manager=memory, db=orm_db)
    return chat_svc, doc_svc, mcp_svc, appr_svc, conv_svc


def configure_page(settings: AppSettings) -> None:
    """Configure Streamlit page metadata before any other UI calls."""
    st.set_page_config(
        page_title=settings.page_title,
        page_icon=settings.page_icon,
        layout="wide",
        initial_sidebar_state="auto",
    )


def on_new_chat(chat_service: Optional[ChatService] = None) -> None:
    """Persist current conversation and transition to a fresh chat thread."""
    start_new_chat()
    if chat_service is not None:
        try:
            chat_service.create_thread(
                user_id=st.session_state.user_id,
                chat_mode=st.session_state.chat_mode,
                thread_id=st.session_state.current_chat_id,
            )
        except Exception as exc:
            logger.warning("Failed to create SQLite thread on new chat: %s", exc)
    st.rerun()


def on_clear_chat(chat_service: Optional[ChatService] = None) -> None:
    """Clear current conversation messages and attachments."""
    thread_id = st.session_state.get("current_chat_id", "")
    user_id = st.session_state.get("user_id", "local_user")
    clear_current_chat()
    if chat_service is not None and thread_id:
        try:
            chat_service.sync_thread_messages(thread_id=thread_id, user_id=user_id, messages=[])
        except Exception as exc:
            logger.warning("Failed to clear persisted thread messages: %s", exc)
    st.rerun()


def create_agent(sidebar_cfg: SidebarConfig, settings: AppSettings) -> AIAgent:
    """Instantiate AIAgent from active sidebar configuration and app settings."""
    return AIAgent(
        text_model=sidebar_cfg.text_model,
        vision_model=sidebar_cfg.vision_model,
        temperature=sidebar_cfg.temperature,
        max_tokens=sidebar_cfg.max_tokens,
        api_key=settings.groq_api_key,
        system_prompt=sidebar_cfg.system_prompt,
        stream_delay_sec=settings.stream_chunk_delay_sec,
    )


def _update_workspace_monitor(
    task: str,
    result: ChatTurnResult,
) -> None:
    st.session_state["agent_workspace_state"] = {
        "task": task,
        "status": result.agent_status,
        "planned_steps": result.planned_steps,
        "tool_executions": result.tool_executions,
        "execution_trace": result.execution_trace,
        "final_result": result.answer,
    }


def _finalize_turn_output(
    result: ChatTurnResult,
    chat_service: ChatService,
    sidebar_cfg: SidebarConfig,
    extra_sources: Optional[Sequence[Dict[str, Any]]] = None,
) -> None:
    if result.pending_approval is not None:
        st.session_state.pending_approval = result.pending_approval
        _update_workspace_monitor(
            st.session_state.messages[-1].get("content", "") if st.session_state.messages else "",
            result,
        )
        return

    st.session_state.pending_approval = None
    merged_sources = list(extra_sources or []) + list(result.sources or [])
    answer_text = result.answer or "I could not generate a response."
    st.write_stream(chat_service.stream_text_tokens(answer_text))
    render_sources_panel(
        sources=merged_sources,
        web_sources=result.web_sources,
        tool_events=result.tool_executions,
        execution_trace=result.execution_trace,
    )

    ai_msg = {
        "role": "assistant",
        "content": answer_text,
        "sources": merged_sources,
        "web_sources": result.web_sources,
        "tool_events": result.tool_executions,
        "execution_trace": result.execution_trace,
        "timestamp": time.time(),
    }
    st.session_state.messages.append(ai_msg)
    chat_service.persist_message(
        thread_id=st.session_state.current_chat_id,
        user_id=sidebar_cfg.user_id,
        message=ai_msg,
        chat_mode=sidebar_cfg.chat_mode,
    )
    _update_workspace_monitor(
        next(
            (m.get("content", "") for m in reversed(st.session_state.messages) if m.get("role") == "user"),
            "",
        ),
        result,
    )


def execute_ai_turn(
    agent: AIAgent,
    has_image: bool = False,
    chat_service: Optional[ChatService] = None,
    sidebar_cfg: Optional[SidebarConfig] = None,
    override_query: Optional[str] = None,
    extra_sources: Optional[Sequence[Dict[str, Any]]] = None,
) -> None:
    """Execute multimodal vision turn or LangGraph Agentic RAG + MCP turn."""
    col_ai, _ = st.columns([9, 1])
    with col_ai:
        st.markdown('<div class="msg-label-ai">QnAGPT</div>', unsafe_allow_html=True)
        with st.chat_message("assistant"):
            try:
                if has_image or chat_service is None or sidebar_cfg is None:
                    if has_image and sidebar_cfg is not None and sidebar_cfg.text_model != sidebar_cfg.vision_model:
                        st.caption(
                            f"Routing image attachment to vision-capable model `{sidebar_cfg.vision_model}`."
                        )
                    with st.spinner("Analyzing..."):
                        stream_gen = agent.stream(st.session_state.messages, has_image=has_image)
                        answer = st.write_stream(stream_gen)
                    if answer:
                        ai_msg = {
                            "role": "assistant",
                            "content": answer,
                            "sources": list(extra_sources or []),
                            "web_sources": [],
                            "tool_events": [],
                            "timestamp": time.time(),
                        }
                        st.session_state.messages.append(ai_msg)
                        if chat_service is not None and sidebar_cfg is not None:
                            chat_service.persist_message(
                                thread_id=st.session_state.current_chat_id,
                                user_id=sidebar_cfg.user_id,
                                message=ai_msg,
                                chat_mode=sidebar_cfg.chat_mode,
                            )
                    return

                last_user = next(
                    (m for m in reversed(st.session_state.messages) if m.get("role") == "user"),
                    None,
                )
                query_text = override_query or str(last_user.get("content", "") if last_user else "")
                max_iter = int(st.session_state.get("cfg_max_iterations", 10))
                has_status_widget = callable(getattr(st, "status", None)) and callable(getattr(st, "empty", None))

                if has_status_widget:
                    with st.status("Processing request...", expanded=True) as status_ui:
                        timeline_slot = st.empty()
                        render_html(
                            render_execution_timeline_html(
                                [{"label": "Understanding request", "detail": "", "state": "in_progress"}]
                            ),
                            target=timeline_slot,
                        )

                        def _on_stage(ev: Dict[str, Any]) -> None:
                            trace = list(ev.get("trace") or [])
                            cur = ev.get("stage") or {}
                            lbl = str(cur.get("label") or "Processing...")
                            status_ui.update(label=lbl)
                            render_html(
                                render_execution_timeline_html(trace),
                                target=timeline_slot,
                            )

                        result = chat_service.run_agentic_turn(
                            user_query=query_text,
                            thread_id=st.session_state.current_chat_id,
                            user_id=sidebar_cfg.user_id,
                            chat_mode=sidebar_cfg.chat_mode,
                            model_name=sidebar_cfg.text_model,
                            temperature=sidebar_cfg.temperature,
                            max_tokens=sidebar_cfg.max_tokens,
                            system_prompt=sidebar_cfg.system_prompt,
                            on_event=_on_stage,
                            max_iterations=max_iter,
                        )
                        total_s = result.execution_trace[-1]["elapsed_s"] if result.execution_trace else 0.0
                        final_state = "error" if result.error and not result.answer else "complete"
                        status_ui.update(
                            label=f"Completed ({len(result.execution_trace)} stages · {total_s:.1f}s)",
                            state=final_state,
                            expanded=False,
                        )
                else:
                    status_box = st.empty()
                    status_box.caption("Understanding your request...")
                    result = chat_service.run_agentic_turn(
                        user_query=query_text,
                        thread_id=st.session_state.current_chat_id,
                        user_id=sidebar_cfg.user_id,
                        chat_mode=sidebar_cfg.chat_mode,
                        model_name=sidebar_cfg.text_model,
                        temperature=sidebar_cfg.temperature,
                        max_tokens=sidebar_cfg.max_tokens,
                        system_prompt=sidebar_cfg.system_prompt,
                        on_activity=lambda msg: status_box.caption(msg),
                        max_iterations=max_iter,
                    )
                    status_box.empty()

                _finalize_turn_output(
                    result,
                    chat_service=chat_service,
                    sidebar_cfg=sidebar_cfg,
                    extra_sources=extra_sources,
                )
            except Exception as exc:
                logger.error("Failed to execute AI turn: %s", exc)
                st.error("Something went wrong while contacting the model.")
                with st.expander("Details"):
                    st.code(str(exc))


def handle_approval_decision(
    approval_id: str,
    approved: bool,
    chat_service: ChatService,
    sidebar_cfg: SidebarConfig,
) -> None:
    """Resume a paused LangGraph execution after user Approve/Reject action."""
    col_ai, _ = st.columns([11, 1])
    with col_ai:
        with st.chat_message("assistant"):
            try:
                if callable(getattr(st, "status", None)) and callable(getattr(st, "empty", None)):
                    with st.status(
                        "Executing approved action..." if approved else "Canceling rejected action...",
                        expanded=True,
                    ) as status_ui:
                        timeline_slot = st.empty()

                        def _on_resume_stage(ev: Dict[str, Any]) -> None:
                            timeline_slot.markdown(
                                render_execution_timeline_html(ev.get("trace") or []),
                                unsafe_allow_html=True,
                            )

                        result = chat_service.resume_approval(
                            thread_id=st.session_state.current_chat_id,
                            user_id=sidebar_cfg.user_id,
                            approval_id=approval_id,
                            approved=approved,
                            model_name=sidebar_cfg.text_model,
                            temperature=sidebar_cfg.temperature,
                            max_tokens=sidebar_cfg.max_tokens,
                            on_event=_on_resume_stage,
                        )
                        status_ui.update(label="Completed", state="complete", expanded=False)
                else:
                    status_box = st.empty()
                    status_box.caption("Resuming approved action..." if approved else "Canceling rejected action...")
                    result = chat_service.resume_approval(
                        thread_id=st.session_state.current_chat_id,
                        user_id=sidebar_cfg.user_id,
                        approval_id=approval_id,
                        approved=approved,
                        model_name=sidebar_cfg.text_model,
                        temperature=sidebar_cfg.temperature,
                        max_tokens=sidebar_cfg.max_tokens,
                        on_activity=lambda msg: status_box.caption(msg),
                    )
                    status_box.empty()
                _finalize_turn_output(
                    result,
                    chat_service=chat_service,
                    sidebar_cfg=sidebar_cfg,
                )
            except Exception as exc:
                st.session_state.pending_approval = None
                st.error(f"Failed resuming workflow: {exc}")
    st.rerun()


def process_pending_prompt(
    agent: AIAgent,
    chat_service: Optional[ChatService] = None,
    sidebar_cfg: Optional[SidebarConfig] = None,
) -> Optional[str]:
    """Handle suggestion clicks and response regeneration triggers."""
    pending = st.session_state.prompt_to_send
    if not pending:
        return None

    st.session_state.prompt_to_send = None
    if pending == "__REGENERATE__":
        if chat_service is not None and sidebar_cfg is not None:
            try:
                chat_service.sync_thread_messages(
                    thread_id=st.session_state.current_chat_id,
                    user_id=sidebar_cfg.user_id,
                    messages=list(st.session_state.messages),
                )
            except Exception as exc:
                logger.warning("Failed syncing thread before regenerate: %s", exc)

        last_user = next(
            (m for m in reversed(st.session_state.messages) if m.get("role") == "user"),
            None,
        )
        has_image = bool(
            last_user and (last_user.get("image_b64") or last_user.get("images"))
        )
        execute_ai_turn(
            agent,
            has_image=has_image,
            chat_service=chat_service,
            sidebar_cfg=sidebar_cfg,
        )
        st.rerun()

    return pending


def submit_user_message(
    agent: AIAgent,
    prompt_text: str,
    chat_service: Optional[ChatService] = None,
    sidebar_cfg: Optional[SidebarConfig] = None,
    images: Optional[Sequence[Dict[str, Any]]] = None,
    documents: Optional[Sequence[Tuple[str, bytes]]] = None,
    doc_action: str = "Index into Knowledge Base",
    document_service: Optional[DocumentService] = None,
    storage: Optional[LocalStorageProvider] = None,
) -> None:
    """Persist user message (with optional multi-image & document attachments) and run AI turn."""
    img_list = list(images or [])
    img_b64 = img_mime = None
    if img_list:
        img_b64 = img_list[0].get("b64")
        img_mime = img_list[0].get("mime")
    elif st.session_state.get("pending_image"):
        img_b64 = st.session_state.pending_image["b64"]
        img_mime = st.session_state.pending_image["mime"]
        img_list = [st.session_state.pending_image]
        st.session_state.pending_image = None
        st.session_state.uploader_nonce = int(st.session_state.get("uploader_nonce", 0)) + 1

    if storage is not None and sidebar_cfg is not None and img_list:
        for img_item in img_list:
            raw_b64 = img_item.get("b64")
            if raw_b64 and not img_item.get("storage_path"):
                try:
                    raw_img_bytes = base64.b64decode(raw_b64)
                    meta = storage.save_file(
                        user_id=sidebar_cfg.user_id,
                        bucket="images",
                        filename=str(img_item.get("name") or "attachment.png"),
                        data=raw_img_bytes,
                    )
                    img_item["storage_path"] = meta.relative_path
                except Exception as exc:
                    logger.warning("Failed persisting uploaded image to LocalStorageProvider: %s", exc)

    doc_files = list(documents or [])
    doc_names = [name for name, _ in doc_files]
    override_query: Optional[str] = None
    extra_sources: List[Dict[str, Any]] = []

    if doc_files and chat_service is not None and sidebar_cfg is not None:
        if doc_action == "Index into Knowledge Base" and document_service is not None:
            results = document_service.ingest_files(doc_files, user_id=sidebar_cfg.user_id)
            for res in results:
                if res.status == "ingested":
                    st.toast(f"Stored & indexed {res.filename} ({res.chunk_count} chunks)")
                elif res.status == "duplicate":
                    st.toast(f"Already indexed: {res.filename}")
                else:
                    st.error(f"Could not index {res.filename}: {res.error}")
        else:
            try:
                inline_ctx, inline_srcs = chat_service.extract_inline_document_context(doc_files)
                extra_sources.extend(inline_srcs)
                override_query = (
                    f"{prompt_text}\n\n"
                    f"=== ATTACHED DOCUMENT CONTEXT (ANALYZE ONLY) ===\n{inline_ctx}"
                )
            except Exception as exc:
                st.error(f"Failed parsing attached document(s): {exc}")

    user_msg = {
        "role": "user",
        "content": prompt_text,
        "image_b64": img_b64,
        "image_mime": img_mime,
        "images": img_list,
        "attached_docs": doc_names,
        "timestamp": time.time(),
    }
    st.session_state.messages.append(user_msg)
    st.session_state.composer_nonce = int(st.session_state.get("composer_nonce", 0)) + 1

    if chat_service is not None and sidebar_cfg is not None:
        try:
            chat_service.persist_message(
                thread_id=st.session_state.current_chat_id,
                user_id=sidebar_cfg.user_id,
                message=user_msg,
                chat_mode=sidebar_cfg.chat_mode,
            )
        except Exception as exc:
            logger.warning("Failed to persist user message: %s", exc)

    render_user_message(
        content=prompt_text,
        image_b64=img_b64,
        image_mime=img_mime or "image/png",
        images=img_list,
        attached_docs=doc_names,
    )
    execute_ai_turn(
        agent,
        has_image=bool(img_list),
        chat_service=chat_service,
        sidebar_cfg=sidebar_cfg,
        override_query=override_query,
        extra_sources=extra_sources,
    )
    st.rerun()


def _clear_authenticated_client_state() -> None:
    """Reset session state upon sign-out or account deletion without deleting cached services."""
    import uuid

    st.session_state.pop("auth_session_token", None)
    st.session_state.user_id = ""
    st.session_state.messages = []
    st.session_state.recent_chats = []
    st.session_state.feedback = {}
    st.session_state.pending_approval = None
    st.session_state.active_page = "Chat"
    st.session_state.current_chat_id = f"conversation_{uuid.uuid4().hex[:12]}"


def main() -> None:
    """Run the QnAGPT Production Multi-User Agentic AI Workspace."""
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_page(settings)

    initialize_session_state()
    apply_styles()

    orm_db, storage, vector_store, auth_svc, user_svc = build_platform_services(settings)
    oauth_err = handle_oauth_callback_params(auth_svc, user_svc)
    auth_session = auth_svc.validate_session(st.session_state.get("auth_session_token"))

    if auth_session is None:
        render_auth_gate(settings, auth_svc, user_svc, oauth_error=oauth_err)
        return

    # Authoritative backend session resolution overrides any frontend state
    st.session_state.user_id = auth_session.user.id
    chat_svc, doc_svc, mcp_svc, appr_svc, conv_svc = build_runtime_services(settings)

    def _handle_sign_out() -> None:
        auth_svc.logout(st.session_state.get("auth_session_token"))
        _clear_authenticated_client_state()

    sidebar_cfg = render_sidebar(
        settings=settings,
        api_key_present=bool(settings.groq_api_key),
        on_new_chat=lambda: on_new_chat(chat_svc),
        on_clear_chat=lambda: on_clear_chat(chat_svc),
        chat_service=chat_svc,
        document_service=doc_svc,
        mcp_service=mcp_svc,
        conversation_service=conv_svc,
        authenticated_user=auth_session.user,
        on_sign_out=_handle_sign_out,
    )
    agent = create_agent(sidebar_cfg, settings)

    page = sidebar_cfg.active_page
    if page == "Agent Workspace":
        render_agent_workspace_page(
            user_id=sidebar_cfg.user_id,
            mcp_service=mcp_svc,
            document_service=doc_svc,
            approval_service=appr_svc,
            on_approval_decision=lambda aid, ok: handle_approval_decision(aid, ok, chat_svc, sidebar_cfg),
        )
        return

    if page == "Knowledge Base":
        render_knowledge_base_page(settings, sidebar_cfg.user_id, doc_svc)
        return

    if page == "MCP Marketplace":
        render_mcp_marketplace_page(settings, sidebar_cfg.user_id, mcp_svc)
        return

    if page == "MCP Servers":
        render_mcp_dashboard(sidebar_cfg.user_id, mcp_svc)
        return

    if page == "Tool Marketplace":
        render_tool_marketplace_page(sidebar_cfg.user_id, mcp_svc)
        return

    if page == "Account":
        render_account_page(
            settings=settings,
            auth_session=auth_session,
            auth_service=auth_svc,
            user_service=user_svc,
            vector_store=vector_store,
            on_signed_out=_clear_authenticated_client_state,
        )
        return

    if page == "Settings":
        render_settings_page(
            settings=settings,
            user_id=sidebar_cfg.user_id,
            on_clear_chat=lambda: on_clear_chat(chat_svc),
            user_service=user_svc,
            document_service=doc_svc,
        )
        return

    # ── Chat Page Header + Conversation Canvas ───────────────────────────────
    active_conv_title = (
        _derive_chat_title(st.session_state.messages)
        if st.session_state.get("messages")
        else "New Conversation"
    )
    render_header(
        sidebar_cfg.text_model,
        chat_mode=sidebar_cfg.chat_mode,
        page_title=page,
        conversation_title=active_conv_title,
        settings=settings,
        on_clear_chat=lambda: on_clear_chat(chat_svc),
    )

    if not st.session_state.messages:
        sug_svc = SuggestionService(
            settings=settings,
            mcp_service=mcp_svc,
            document_service=doc_svc,
            conversation_service=conv_svc,
        )
        force_ref = bool(st.session_state.pop("_force_refresh_suggestions", False))
        suggestions = sug_svc.get_suggestions(
            user_id=sidebar_cfg.user_id,
            current_chat_id=st.session_state.current_chat_id,
            model_name=sidebar_cfg.text_model,
            cache_store=st.session_state,
            force_refresh=force_ref,
            allow_llm=force_ref,
        )

        def _refresh_sugs() -> None:
            st.session_state["_force_refresh_suggestions"] = True
            st.rerun()

        render_welcome_screen(suggestions=suggestions, on_refresh=_refresh_sugs)
    else:
        render_chat_history(
            messages=st.session_state.messages,
            search_query=st.session_state.search_query,
            on_feedback=lambda rating, body: conv_svc.record_feedback(
                user_id=sidebar_cfg.user_id,
                conversation_id=st.session_state.current_chat_id,
                rating=rating,
                message_content=body,
            ),
        )

    pending_appr = st.session_state.get("pending_approval")
    if pending_appr:
        render_approval_card(
            pending_appr,
            on_decision=lambda aid, ok: handle_approval_decision(aid, ok, chat_svc, sidebar_cfg),
            key_prefix="chat_appr",
        )

    triggered_prompt = process_pending_prompt(
        agent,
        chat_service=chat_svc,
        sidebar_cfg=sidebar_cfg,
    )
    composer_sub = render_chat_composer(
        settings=settings,
        active_model=sidebar_cfg.text_model,
        disabled=bool(pending_appr),
    )

    if composer_sub is not None:
        submit_user_message(
            agent,
            prompt_text=composer_sub.text,
            chat_service=chat_svc,
            sidebar_cfg=sidebar_cfg,
            images=composer_sub.images,
            documents=composer_sub.documents,
            doc_action=composer_sub.doc_action,
            document_service=doc_svc,
            storage=storage,
        )
    elif triggered_prompt:
        submit_user_message(
            agent,
            prompt_text=triggered_prompt,
            chat_service=chat_svc,
            sidebar_cfg=sidebar_cfg,
            document_service=doc_svc,
            storage=storage,
        )


if __name__ == "__main__":
    main()
