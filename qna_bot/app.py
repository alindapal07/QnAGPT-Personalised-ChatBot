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

from qna_bot.config.settings import AppSettings, get_settings  # noqa: E402
from qna_bot.core.logging_config import configure_logging, get_logger  # noqa: E402
from qna_bot.services.agent_service import AIAgent  # noqa: E402
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
    render_user_message,
    render_welcome_screen,
)

logger = get_logger(__name__)


def configure_page(settings: AppSettings) -> None:
    """Configure Streamlit page metadata before any other UI calls."""
    st.set_page_config(
        page_title=settings.page_title,
        page_icon=settings.page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )


def on_new_chat() -> None:
    """Persist current conversation and transition to a fresh chat."""
    start_new_chat()
    st.rerun()


def on_clear_chat() -> None:
    """Clear current conversation messages and attachments."""
    clear_current_chat()
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


def execute_ai_turn(agent: AIAgent, has_image: bool = False) -> None:
    """Stream an assistant response and append it to session history."""
    col_ai, _ = st.columns([4, 1])
    with col_ai:
        st.markdown('<div class="msg-label-ai">Assistant</div>', unsafe_allow_html=True)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Generating..."):
                    stream_gen = agent.stream(st.session_state.messages, has_image=has_image)
                    answer = st.write_stream(stream_gen)
                if answer:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
            except Exception as exc:
                logger.error("Failed to execute AI turn: %s", exc)
                st.error("Something went wrong while contacting the model.")
                with st.expander("Details"):
                    st.code(str(exc))


def process_pending_prompt(agent: AIAgent) -> Optional[str]:
    """Handle suggestion clicks and response regeneration triggers."""
    pending = st.session_state.prompt_to_send
    if not pending:
        return None

    st.session_state.prompt_to_send = None
    if pending == "__REGENERATE__":
        last_user = next(
            (m for m in reversed(st.session_state.messages) if m.get("role") == "user"),
            None,
        )
        has_image = bool(last_user and last_user.get("image_b64"))
        execute_ai_turn(agent, has_image=has_image)
        st.rerun()

    return pending


def submit_user_message(agent: AIAgent, prompt_text: str) -> None:
    """Persist user message (with optional image), render it, and run AI turn."""
    img_b64 = img_mime = None
    if st.session_state.pending_image:
        img_b64 = st.session_state.pending_image["b64"]
        img_mime = st.session_state.pending_image["mime"]
        st.session_state.pending_image = None
        st.session_state.uploader_nonce = int(st.session_state.get("uploader_nonce", 0)) + 1

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt_text,
            "image_b64": img_b64,
            "image_mime": img_mime,
            "timestamp": time.time(),
        }
    )

    render_user_message(
        content=prompt_text,
        image_b64=img_b64,
        image_mime=img_mime or "image/png",
    )
    execute_ai_turn(agent, has_image=bool(img_b64))
    st.rerun()


def main() -> None:
    """Run the Streamlit QnA Chatbot application workflow."""
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_page(settings)

    initialize_session_state()
    apply_styles()

    sidebar_cfg = render_sidebar(
        settings=settings,
        api_key_present=bool(settings.groq_api_key),
        on_new_chat=on_new_chat,
        on_clear_chat=on_clear_chat,
    )
    agent = create_agent(sidebar_cfg, settings)

    render_header(sidebar_cfg.text_model)

    if not st.session_state.messages:
        render_welcome_screen()
    else:
        render_chat_history(
            messages=st.session_state.messages,
            search_query=st.session_state.search_query,
        )

    render_image_upload_section(settings)

    triggered_prompt = process_pending_prompt(agent)
    user_query = st.chat_input("Ask anything...")
    final_prompt = user_query or triggered_prompt

    send_image_only = (
        bool(st.session_state.pending_image)
        and not final_prompt
        and st.button("Send image", key="btn_send_img_only")
    )

    if final_prompt or send_image_only:
        prompt_text = final_prompt or "Please analyze this image."
        submit_user_message(agent, prompt_text)


if __name__ == "__main__":
    main()
