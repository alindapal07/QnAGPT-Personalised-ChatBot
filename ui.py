"""
ui.py — Streamlit UI rendering components and layout helpers.
Contains ALL presentation rendering.
Contains NO AI / LangChain / Groq model implementation.
"""
import io
import json
import base64
from typing import Any, Callable, Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from config import (
    ALLOWED_IMAGE_TYPES,
    AVAILABLE_TEXT_MODELS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_VISION_MODEL,
    MAX_IMAGE_SIZE_MB,
)


# ── Image processing ──────────────────────────────────────────────────────────

def process_uploaded_image(uploaded_file) -> Optional[Dict[str, Any]]:
    """Validate and convert uploaded image into base64 format."""
    if uploaded_file is None:
        return None
    try:
        file_bytes = uploaded_file.getvalue()
        if len(file_bytes) / (1024 * 1024) > MAX_IMAGE_SIZE_MB:
            st.error(f"Image exceeds {MAX_IMAGE_SIZE_MB}MB limit.")
            return None

        pil_image = Image.open(io.BytesIO(file_bytes))
        pil_image.verify()
        pil_image = Image.open(io.BytesIO(file_bytes))

        fmt = (pil_image.format or "PNG").lower()
        if fmt == "jpeg":
            fmt = "jpg"
        mime = f"image/{'jpeg' if fmt == 'jpg' else fmt}"
        b64 = base64.b64encode(file_bytes).decode("utf-8")

        return {
            "name": uploaded_file.name,
            "b64": b64,
            "mime": mime,
            "format": fmt,
            "width": pil_image.width,
            "height": pil_image.height,
        }
    except Exception as e:
        st.error(f"Invalid or corrupted image: {e}")
        return None


# ── Copy button component ─────────────────────────────────────────────────────

def render_copy_button(text: str, button_id: str) -> None:
    """Render a lightweight JS-powered copy button that shows 'Copied' for 1.8s."""
    escaped = json.dumps(text)
    html_code = f"""
    <div style="margin: 0; padding: 0;">
      <button id="btn_{button_id}" onclick="copyText_{button_id}()" style="
        background: transparent;
        border: 1px solid #d1d5db;
        border-radius: 5px;
        color: #4b5563;
        font-family: inherit;
        font-size: 11.5px;
        padding: 2px 8px;
        cursor: pointer;
        line-height: 1.4;
        transition: background-color 0.15s, color 0.15s, border-color 0.15s;
      ">Copy</button>
      <script>
        (function() {{
          const btn = document.getElementById("btn_{button_id}");
          function applyThemeStyle() {{
            let isDark = false;
            try {{
              if (window.parent && window.parent.document) {{
                const doc = window.parent.document;
                const dt = doc.documentElement.getAttribute('data-theme') || (doc.body && doc.body.getAttribute('data-theme'));
                if (dt === 'dark') {{ isDark = true; }}
                else if (dt === 'light') {{ isDark = false; }}
                else {{
                  const app = doc.querySelector('.stApp') || doc.body;
                  const bg = window.parent.getComputedStyle(app).backgroundColor;
                  const m = bg.match(/\\d+/g);
                  if (m && m.length >= 3) {{
                    const lum = 0.299 * parseInt(m[0]) + 0.587 * parseInt(m[1]) + 0.114 * parseInt(m[2]);
                    isDark = lum < 128;
                  }}
                }}
              }}
            }} catch(e) {{
              isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
            }}

            if (isDark) {{
              btn.style.borderColor = "#303642";
              btn.style.color = "#CBD5E1";
            }} else {{
              btn.style.borderColor = "#E5E7EB";
              btn.style.color = "#4B5563";
            }}
          }}
          applyThemeStyle();
          setTimeout(applyThemeStyle, 100);
        }})();

        function copyText_{button_id}() {{
          const val = {escaped};
          const btn = document.getElementById("btn_{button_id}");
          if (navigator.clipboard && navigator.clipboard.writeText) {{
            navigator.clipboard.writeText(val).then(() => {{
              btn.innerText = "Copied";
              setTimeout(() => {{ btn.innerText = "Copy"; }}, 1800);
            }}).catch(() => fallback_{button_id}(val, btn));
          }} else {{
            fallback_{button_id}(val, btn);
          }}
        }}
        function fallback_{button_id}(val, btn) {{
          const ta = document.createElement("textarea");
          ta.value = val;
          ta.style.position = "fixed";
          ta.style.opacity = "0";
          document.body.appendChild(ta);
          ta.select();
          document.execCommand("copy");
          document.body.removeChild(ta);
          btn.innerText = "Copied";
          setTimeout(() => {{ btn.innerText = "Copy"; }}, 1800);
        }}
      </script>
    </div>
    """
    components.html(html_code, height=28)


# ── Header ────────────────────────────────────────────────────────────────────

def render_header(model_name: str) -> None:
    """Render subtle enterprise top header."""
    st.markdown(
        f"""
        <div class="app-header">
            <span class="app-header-title">AI Assistant</span>
            <span class="app-header-meta">
                Model:&nbsp;<code>{model_name}</code>&nbsp;&nbsp;
                Powered by Groq
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Welcome Screen ────────────────────────────────────────────────────────────

def render_welcome_screen() -> None:
    """Render welcome screen with suggestion prompts."""
    st.markdown(
        """
        <div class="welcome-wrap">
            <div class="welcome-title">AI Assistant</div>
            <div class="welcome-sub">
                Ask questions, analyze information, or upload an image.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    suggestions = [
        ("Explain LangChain", "Can you explain LangChain architecture, components, and how chains work?"),
        ("Help debug Python", "What are best practices for debugging and error handling in Python?"),
        ("How does Vision AI work?", "How do multimodal vision models process and interpret images alongside text?"),
        ("Write a FastAPI template", "Show me an idiomatic FastAPI project template with Pydantic validation."),
    ]
    for i, (label, prompt) in enumerate(suggestions):
        col = c1 if i % 2 == 0 else c2
        with col:
            if st.button(label, key=f"sug_{i}", use_container_width=True):
                st.session_state.prompt_to_send = prompt
                st.rerun()


# ── Messages (Right = User, Left = AI) ────────────────────────────────────────

def render_user_message(content: str, image_b64: Optional[str] = None, image_mime: str = "image/png") -> None:
    """Render user message on the RIGHT side."""
    col_spacer, col_user = st.columns([1, 4])
    with col_user:
        st.markdown('<div class="msg-label-user">You</div>', unsafe_allow_html=True)
        if image_b64:
            img_tag = (
                f'<img src="data:{image_mime};base64,{image_b64}" '
                f'style="max-width:100%;max-height:260px;border-radius:6px;display:block;margin-bottom:8px;border:1px solid var(--border);" />'
            )
            st.markdown(f'<div style="text-align:right">{img_tag}</div>', unsafe_allow_html=True)
        if content.strip():
            with st.chat_message("user"):
                st.markdown(content)


def render_ai_message(content: str, index: int, is_last_ai: bool = False) -> None:
    """Render AI message on the LEFT side with compact action buttons."""
    col_ai, col_spacer = st.columns([4, 1])
    with col_ai:
        st.markdown('<div class="msg-label-ai">Assistant</div>', unsafe_allow_html=True)
        with st.chat_message("assistant"):
            st.markdown(content)

        # Action row
        a1, a2, a3, a4, _ = st.columns([1, 1.2, 1, 1.4, 4])
        with a1:
            if st.button("Helpful", key=f"helpful_{index}"):
                st.session_state.feedback[index] = "helpful"
                st.toast("Marked as helpful.")
        with a2:
            if st.button("Not helpful", key=f"unhelpful_{index}"):
                st.session_state.feedback[index] = "unhelpful"
                st.toast("Feedback recorded.")
        with a3:
            render_copy_button(content, f"ai_{index}")
        with a4:
            if is_last_ai:
                if st.button("Regenerate", key=f"regen_{index}"):
                    st.session_state.messages.pop()
                    st.session_state.prompt_to_send = "__REGENERATE__"
                    st.rerun()


def render_chat_history(messages: List[Dict[str, Any]], search_query: str = "") -> None:
    """Render all past messages in the conversation."""
    ai_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    last_ai_idx = ai_indices[-1] if ai_indices else -1

    for idx, msg in enumerate(messages):
        content = msg.get("content", "")
        if search_query and search_query.lower() not in content.lower():
            continue

        if msg.get("role") == "user":
            render_user_message(
                content=content,
                image_b64=msg.get("image_b64"),
                image_mime=msg.get("image_mime", "image/png"),
            )
        elif msg.get("role") == "assistant":
            render_ai_message(
                content=content,
                index=idx,
                is_last_ai=(idx == last_ai_idx),
            )


# ── Image Attachment UI ───────────────────────────────────────────────────────

def render_image_upload_section() -> None:
    """Render compact, theme-synchronized image attachment section."""
    has_image = bool(st.session_state.pending_image)
    with st.expander("Attach image", expanded=has_image):
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=ALLOWED_IMAGE_TYPES,
            key="img_file_uploader",
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            processed = process_uploaded_image(uploaded_file)
            if processed:
                st.session_state.pending_image = processed

        if st.session_state.pending_image:
            img = st.session_state.pending_image
            st.markdown(
                f'<div class="img-preview-bar">'
                f'<span>Attached: <strong>{img["name"]}</strong> ({img["width"]}x{img["height"]})</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            col_p, col_r = st.columns([3, 1])
            with col_p:
                st.image(base64.b64decode(img["b64"]), width=180)
            with col_r:
                if st.button("Remove", key="btn_remove_pending_img"):
                    st.session_state.pending_image = None
                    st.rerun()


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(
    api_key_present: bool,
    on_new_chat: Callable[[], None],
    on_clear_chat: Callable[[], None],
) -> Dict[str, Any]:
    """Render minimal enterprise sidebar with configuration options."""
    with st.sidebar:
        st.markdown('<div class="sb-brand">AI Assistant</div>', unsafe_allow_html=True)

        if st.button("+ New chat", use_container_width=True):
            on_new_chat()

        st.markdown('<hr class="sb-divider"/>', unsafe_allow_html=True)

        # Connection status
        if api_key_present:
            st.markdown('<div class="sb-status"><span class="dot-ok"></span>Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="sb-status"><span class="dot-err"></span>API key missing</div>', unsafe_allow_html=True)
            st.caption("Set `GROQ_API_KEY` in your `.env` file.")

        # Search
        st.markdown('<div class="sb-section-label">Search</div>', unsafe_allow_html=True)
        search_query = st.text_input(
            "Search",
            placeholder="Search conversations...",
            label_visibility="collapsed",
            key="sb_search_input",
        )
        st.session_state.search_query = search_query.strip()

        # Recent chats
        if st.session_state.recent_chats:
            st.markdown('<div class="sb-section-label">Recent</div>', unsafe_allow_html=True)
            for chat in st.session_state.recent_chats[:6]:
                col_t, col_b = st.columns([5, 1])
                with col_t:
                    st.markdown(f'<div class="sb-chat-title">{chat["title"]}</div>', unsafe_allow_html=True)
                with col_b:
                    if st.button("Open", key=f"restore_{chat['id']}"):
                        st.session_state.messages = list(chat["messages"])
                        st.session_state.current_chat_id = chat["id"]
                        st.rerun()

        st.markdown('<hr class="sb-divider"/>', unsafe_allow_html=True)

        # Settings
        selected_model = AVAILABLE_TEXT_MODELS[0]
        temperature = DEFAULT_TEMPERATURE
        max_tokens = DEFAULT_MAX_TOKENS
        system_prompt = DEFAULT_SYSTEM_PROMPT

        with st.expander("Settings", expanded=False):
            st.markdown('<div class="sb-section-label">Model</div>', unsafe_allow_html=True)
            selected_model = st.selectbox(
                "Text model",
                options=AVAILABLE_TEXT_MODELS,
                index=0,
                label_visibility="collapsed",
            )
            st.caption(f"Vision model: `{DEFAULT_VISION_MODEL}` (auto-routed for images)")

            st.markdown('<div class="sb-section-label">Temperature</div>', unsafe_allow_html=True)
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=DEFAULT_TEMPERATURE,
                step=0.05,
                label_visibility="collapsed",
            )

            st.markdown('<div class="sb-section-label">Max tokens</div>', unsafe_allow_html=True)
            max_tokens = st.slider(
                "Max tokens",
                min_value=256,
                max_value=4096,
                value=DEFAULT_MAX_TOKENS,
                step=256,
                label_visibility="collapsed",
            )

            with st.expander("System prompt", expanded=False):
                system_prompt = st.text_area(
                    "System instructions",
                    value=DEFAULT_SYSTEM_PROMPT,
                    height=90,
                    label_visibility="collapsed",
                )

        # Export
        if st.session_state.messages:
            st.markdown('<div class="sb-section-label">Export</div>', unsafe_allow_html=True)
            json_export = json.dumps(
                {"chat_id": st.session_state.current_chat_id, "messages": st.session_state.messages},
                indent=2,
            )
            md_lines = ["# Chat Export\n"]
            for m in st.session_state.messages:
                speaker = "User" if m["role"] == "user" else "Assistant"
                md_lines.append(f"### {speaker}\n{m.get('content', '')}\n")
            md_export = "\n".join(md_lines)
            txt_export = "\n".join(f"{m['role'].upper()}: {m.get('content', '')}" for m in st.session_state.messages)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.download_button("JSON", data=json_export, file_name="chat.json", mime="application/json", use_container_width=True)
            with c2:
                st.download_button("MD", data=md_export, file_name="chat.md", mime="text/markdown", use_container_width=True)
            with c3:
                st.download_button("TXT", data=txt_export, file_name="chat.txt", mime="text/plain", use_container_width=True)

        # Clear chat
        st.markdown('<hr class="sb-divider"/>', unsafe_allow_html=True)
        if not st.session_state.confirm_clear:
            if st.button("Clear conversation", use_container_width=True):
                st.session_state.confirm_clear = True
                st.rerun()
        else:
            st.warning("Clear this conversation?")
            cy, cn = st.columns(2)
            with cy:
                if st.button("Clear", type="primary", use_container_width=True):
                    on_clear_chat()
            with cn:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.confirm_clear = False
                    st.rerun()

    return {
        "text_model": selected_model,
        "vision_model": DEFAULT_VISION_MODEL,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "system_prompt": system_prompt,
    }

