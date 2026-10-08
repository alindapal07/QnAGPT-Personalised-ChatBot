"""
app.py — Application entry point and orchestrator.
Orchestrates:
1. Config & environment
2. Page setup & theme-aware styling
3. Session state management
4. AI Agent interaction
5. UI layout & stream response handling
"""
import time
import uuid
import streamlit as st

from config import PAGE_TITLE, PAGE_ICON, GROQ_API_KEY
from styles import apply_styles
from agent import AIAgent
from ui import (
    render_header,
    render_sidebar,
    render_welcome_screen,
    render_chat_history,
    render_user_message,
    render_image_upload_section,
)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session State Initialization ──────────────────────────────────────────────
def initialize_session_state() -> None:
    defaults = {
        "messages": [],
        "current_chat_id": str(uuid.uuid4()),
        "recent_chats": [],
        "pending_image": None,
        "feedback": {},
        "confirm_clear": False,
        "search_query": "",
        "prompt_to_send": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def save_current_chat_to_recent() -> None:
    if not st.session_state.messages:
        return
    first_user = next((m["content"] for m in st.session_state.messages if m["role"] == "user"), None)
    title = (first_user[:36] + "...") if first_user and len(first_user) > 36 else (first_user or "Conversation")
    if not any(c.get("id") == st.session_state.current_chat_id for c in st.session_state.recent_chats):
        st.session_state.recent_chats.insert(0, {
            "id": st.session_state.current_chat_id,
            "title": title,
            "messages": list(st.session_state.messages),
            "timestamp": time.strftime("%b %d, %H:%M"),
        })


def on_new_chat() -> None:
    save_current_chat_to_recent()
    st.session_state.messages = []
    st.session_state.pending_image = None
    st.session_state.current_chat_id = str(uuid.uuid4())
    st.session_state.confirm_clear = False
    st.rerun()


def on_clear_chat() -> None:
    st.session_state.messages = []
    st.session_state.pending_image = None
    st.session_state.confirm_clear = False
    st.rerun()


# ── AI Turn Execution ─────────────────────────────────────────────────────────
def execute_ai_turn(agent: AIAgent, has_image: bool = False) -> None:
    """Execute AI streaming turn and save result to conversation history."""
    col_ai, _ = st.columns([4, 1])
    with col_ai:
        st.markdown('<div class="msg-label-ai">Assistant</div>', unsafe_allow_html=True)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Generating..."):
                    stream_gen = agent.stream(st.session_state.messages, has_image=has_image)
                    answer = st.write_stream(stream_gen)
                if answer:
                    st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error("Something went wrong while contacting the model.")
                with st.expander("Details"):
                    st.code(str(e))


# ── Main Orchestration ────────────────────────────────────────────────────────
def main() -> None:
    # 1. State setup
    initialize_session_state()

    # 2. Dynamic theme styling (Light, Dark, Auto)
    apply_styles()

    # 3. Sidebar rendering
    config = render_sidebar(
        api_key_present=bool(GROQ_API_KEY),
        on_new_chat=on_new_chat,
        on_clear_chat=on_clear_chat,
    )

    # 4. Instantiate Agent
    agent = AIAgent(
        text_model=config["text_model"],
        vision_model=config["vision_model"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        api_key=GROQ_API_KEY,
        system_prompt=config["system_prompt"],
    )

    # 5. Header
    render_header(config["text_model"])

    # 6. Chat or Welcome Screen
    if not st.session_state.messages:
        render_welcome_screen()
    else:
        render_chat_history(
            messages=st.session_state.messages,
            search_query=st.session_state.search_query,
        )

    # 7. Image Upload Section (Theme-synchronized)
    render_image_upload_section()

    # 8. Programmatic Prompts (Suggestions / Regeneration)
    triggered_prompt = None
    if st.session_state.prompt_to_send:
        if st.session_state.prompt_to_send == "__REGENERATE__":
            st.session_state.prompt_to_send = None
            last_user = next((m for m in reversed(st.session_state.messages) if m["role"] == "user"), None)
            has_image = bool(last_user and last_user.get("image_b64"))
            execute_ai_turn(agent, has_image=has_image)
            st.rerun()
        else:
            triggered_prompt = st.session_state.prompt_to_send
            st.session_state.prompt_to_send = None

    # 9. Bottom Chat Input
    user_query = st.chat_input("Ask anything...")
    final_prompt = user_query or triggered_prompt

    send_image_only = (
        st.session_state.pending_image
        and not final_prompt
        and st.button("Send image", key="btn_send_img_only")
    )

    if final_prompt or send_image_only:
        prompt_text = final_prompt if final_prompt else "Please analyze this image."
        img_b64 = img_mime = None

        if st.session_state.pending_image:
            img_b64 = st.session_state.pending_image["b64"]
            img_mime = st.session_state.pending_image["mime"]
            st.session_state.pending_image = None

        st.session_state.messages.append({
            "role": "user",
            "content": prompt_text,
            "image_b64": img_b64,
            "image_mime": img_mime,
            "timestamp": time.time(),
        })

        render_user_message(content=prompt_text, image_b64=img_b64, image_mime=img_mime or "image/png")
        execute_ai_turn(agent, has_image=bool(img_b64))
        st.rerun()


if __name__ == "__main__":
    main()
