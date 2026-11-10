"""Application entry point and Streamlit orchestration."""

from pathlib import Path
import sys
import time
from typing import Optional

# Ensure repository root is importable when launched as `streamlit run qna_bot/app.py`
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402

from qna_bot.agent.memory import ConversationMemoryManager  # noqa: E402
from qna_bot.config.settings import AppSettings, get_settings  # noqa: E402
from qna_bot.core.logging_config import configure_logging, get_logger  # noqa: E402
from qna_bot.rag.vector_store import FAISSVectorStoreManager  # noqa: E402
from qna_bot.services.agent_service import AIAgent  # noqa: E402
from qna_bot.services.chat_service import ChatService  # noqa: E402
from qna_bot.services.document_service import DocumentService  # noqa: E402
from qna_bot.services.session_service import (  # noqa: E402
    clear_current_chat,
    initialize_session_state,
    start_new_chat,
)
from qna_bot.ui import (  # noqa: E402
    SidebarConfig,
    apply_styles,
    render_chat_history,
    render_header,
    render_image_upload_section,
    render_sidebar,
    render_sources_panel,
    render_user_message,
    render_welcome_screen,
)

logger = get_logger(__name__)


@st.cache_resource
def _get_shared_memory(db_path: str) -> ConversationMemoryManager:
    return ConversationMemoryManager(db_path=Path(db_path))


@st.cache_resource
def _get_shared_vector_store(vector_dir: str) -> FAISSVectorStoreManager:
    return FAISSVectorStoreManager(base_dir=Path(vector_dir))


def build_runtime_services(settings: AppSettings) -> tuple[ChatService, DocumentService]:
    """Instantiate ChatService and DocumentService backed by cached SQLite & FAISS managers."""
    memory = _get_shared_memory(str(settings.sqlite_db_path))
    vector_store = _get_shared_vector_store(str(settings.vector_store_dir))
    return (
        ChatService(settings=settings, memory_manager=memory, vector_store=vector_store),
        DocumentService(settings=settings, vector_store=vector_store),
    )


def configure_page(settings: AppSettings) -> None:
    """Configure Streamlit page metadata before any other UI calls."""
    st.set_page_config(
        page_title=settings.page_title,
        page_icon=settings.page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
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


def execute_ai_turn(
    agent: AIAgent,
    has_image: bool = False,
    chat_service: Optional[ChatService] = None,
    sidebar_cfg: Optional[SidebarConfig] = None,
) -> None:
    """Execute either multimodal vision turn or LangGraph Agentic RAG turn and persist result."""
    col_ai, _ = st.columns([4, 1])
    with col_ai:
        st.markdown('<div class="msg-label-ai">Assistant</div>', unsafe_allow_html=True)
        with st.chat_message("assistant"):
            try:
                if has_image or chat_service is None or sidebar_cfg is None:
                    with st.spinner("Generating..."):
                        stream_gen = agent.stream(st.session_state.messages, has_image=has_image)
                        answer = st.write_stream(stream_gen)
                    if answer:
                        ai_msg = {
                            "role": "assistant",
                            "content": answer,
                            "sources": [],
                            "web_sources": [],
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
                query_text = str(last_user.get("content", "") if last_user else "")
                status_box = st.empty()
                status_box.caption("Analyzing your question...")

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
                )
                status_box.empty()

                answer_text = result.answer or "I could not generate a response."
                st.write_stream(chat_service.stream_text_tokens(answer_text))
                render_sources_panel(sources=result.sources, web_sources=result.web_sources)

                ai_msg = {
                    "role": "assistant",
                    "content": answer_text,
                    "sources": result.sources,
                    "web_sources": result.web_sources,
                    "timestamp": time.time(),
                }
                st.session_state.messages.append(ai_msg)
                chat_service.persist_message(
                    thread_id=st.session_state.current_chat_id,
                    user_id=sidebar_cfg.user_id,
                    message=ai_msg,
                    chat_mode=sidebar_cfg.chat_mode,
                )
            except Exception as exc:
                logger.error("Failed to execute AI turn: %s", exc)
                st.error("Something went wrong while contacting the model.")
                with st.expander("Details"):
                    st.code(str(exc))


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
        has_image = bool(last_user and last_user.get("image_b64"))
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
) -> None:
    """Persist user message (with optional image), render it, and run AI turn."""
    img_b64 = img_mime = None
    if st.session_state.pending_image:
        img_b64 = st.session_state.pending_image["b64"]
        img_mime = st.session_state.pending_image["mime"]
        st.session_state.pending_image = None
        st.session_state.uploader_nonce = int(st.session_state.get("uploader_nonce", 0)) + 1

    user_msg = {
        "role": "user",
        "content": prompt_text,
        "image_b64": img_b64,
        "image_mime": img_mime,
        "timestamp": time.time(),
    }
    st.session_state.messages.append(user_msg)

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
    )
    execute_ai_turn(
        agent,
        has_image=bool(img_b64),
        chat_service=chat_service,
        sidebar_cfg=sidebar_cfg,
    )
    st.rerun()


def main() -> None:
    """Run the Streamlit QnAGPT Agentic RAG Chatbot workflow."""
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_page(settings)

    initialize_session_state()
    apply_styles()

    chat_service, document_service = build_runtime_services(settings)

    sidebar_cfg = render_sidebar(
        settings=settings,
        api_key_present=bool(settings.groq_api_key),
        on_new_chat=lambda: on_new_chat(chat_service),
        on_clear_chat=lambda: on_clear_chat(chat_service),
        chat_service=chat_service,
        document_service=document_service,
    )
    agent = create_agent(sidebar_cfg, settings)

    render_header(sidebar_cfg.text_model, chat_mode=sidebar_cfg.chat_mode)

    if not st.session_state.messages:
        render_welcome_screen()
    else:
        render_chat_history(
            messages=st.session_state.messages,
            search_query=st.session_state.search_query,
        )

    render_image_upload_section(settings)

    triggered_prompt = process_pending_prompt(
        agent,
        chat_service=chat_service,
        sidebar_cfg=sidebar_cfg,
    )
    user_query = st.chat_input("Ask anything...")
    final_prompt = user_query or triggered_prompt

    send_image_only = (
        bool(st.session_state.pending_image)
        and not final_prompt
        and st.button("Send image", key="btn_send_img_only")
    )

    if final_prompt or send_image_only:
        prompt_text = final_prompt or "Please analyze this image."
        submit_user_message(
            agent,
            prompt_text,
            chat_service=chat_service,
            sidebar_cfg=sidebar_cfg,
        )


if __name__ == "__main__":
    main()
