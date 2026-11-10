"""Reusable Streamlit UI components (header, welcome screen, copy button, image uploader)."""

import base64
import html
import json

import streamlit as st

from qna_bot.config.settings import AppSettings
from qna_bot.core.exceptions import ImageValidationError
from qna_bot.services.image_service import validate_and_encode_image

DEFAULT_SUGGESTIONS = (
    (
        "Explain LangChain",
        "Can you explain LangChain architecture, components, and how chains work?",
    ),
    (
        "Help debug Python",
        "What are best practices for debugging and error handling in Python?",
    ),
    (
        "How does Vision AI work?",
        "How do multimodal vision models process and interpret images alongside text?",
    ),
    (
        "Write a FastAPI template",
        "Show me an idiomatic FastAPI project template with Pydantic validation.",
    ),
)


def render_copy_button(text: str, button_id: str) -> None:
    """Render a lightweight JS-powered copy button that shows 'Copied' for 1.8s."""
    safe_json = json.dumps(text).replace("</", "<\\/")
    html_code = f"""
    <style>html, body {{ margin: 0; padding: 0; overflow: hidden; background: transparent; }}</style>
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
                    const lum = 0.299 * parseInt(m[0], 10) + 0.587 * parseInt(m[1], 10) + 0.114 * parseInt(m[2], 10);
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
          const val = {safe_json};
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
    if hasattr(st, "iframe"):
        st.iframe(html_code, height=28, alt="Copy assistant response")
    else:
        import streamlit.components.v1 as components

        components.html(html_code, height=28)


def render_header(model_name: str) -> None:
    """Render subtle enterprise top header."""
    escaped_model = html.escape(model_name)
    st.markdown(
        f"""
        <div class="app-header">
            <span class="app-header-title">AI Assistant</span>
            <span class="app-header-meta">
                Model:&nbsp;<code>{escaped_model}</code>&nbsp;&nbsp;
                Powered by Groq
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_welcome_screen() -> None:
    """Render welcome screen with starter prompt suggestions."""
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

    col_left, col_right = st.columns(2)
    for idx, (label, prompt) in enumerate(DEFAULT_SUGGESTIONS):
        col = col_left if idx % 2 == 0 else col_right
        with col:
            if st.button(label, key=f"sug_{idx}", use_container_width=True):
                st.session_state.prompt_to_send = prompt
                st.rerun()


def render_image_upload_section(settings: AppSettings) -> None:
    """Render compact, theme-synchronized image attachment section."""
    has_image = bool(st.session_state.pending_image)
    nonce = st.session_state.get("uploader_nonce", 0)

    with st.expander("Attach image", expanded=has_image):
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=list(settings.allowed_image_types),
            key=f"img_file_uploader_{nonce}",
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            try:
                st.session_state.pending_image = validate_and_encode_image(
                    file_bytes=uploaded_file.getvalue(),
                    filename=uploaded_file.name,
                    max_size_mb=settings.max_image_size_mb,
                    allowed_types=settings.allowed_image_types,
                )
            except ImageValidationError as exc:
                st.error(str(exc))

        if st.session_state.pending_image:
            img = st.session_state.pending_image
            safe_name = html.escape(img["name"])
            st.markdown(
                f'<div class="img-preview-bar">'
                f'<span>Attached: <strong>{safe_name}</strong> ({img["width"]}x{img["height"]})</span>'
                f"</div>",
                unsafe_allow_html=True,
            )
            col_preview, col_remove = st.columns([3, 1])
            with col_preview:
                st.image(base64.b64decode(img["b64"]), width=180)
            with col_remove:
                if st.button("Remove", key="btn_remove_pending_img"):
                    st.session_state.pending_image = None
                    st.session_state.uploader_nonce = nonce + 1
                    st.rerun()
