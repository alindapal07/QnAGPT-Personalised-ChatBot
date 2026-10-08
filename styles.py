"""
styles.py — Centralized semantic theme system and styling for Streamlit.
Supports Light, Dark, and Auto (system) themes seamlessly.
NO AI logic, NO model calls, NO conversation state.
"""
import streamlit as st

THEME_CSS = """
<style>
/* ═══════════════════════════════════════════════════════════════════════════
   1. SEMANTIC THEME VARIABLES
   ═══════════════════════════════════════════════════════════════════════════ */

/* Default / LIGHT THEME */
:root,
[data-theme="light"],
body[data-theme="light"] {
    --app-background:      #FFFFFF;
    --sidebar-background:  #F8F9FB;
    --surface:             #FFFFFF;
    --surface-secondary:   #F3F4F6;
    --surface-hover:       #F3F4F6;
    --border:              #E5E7EB;
    --text-primary:        #111827;
    --text-secondary:      #4B5563;
    --text-muted:          #6B7280;
    --input-background:    #FFFFFF;
    --input-text:          #111827;
    --input-placeholder:   #6B7280;
    --accent:              #2563EB;
    --accent-hover:        #1D4ED8;
    --code-background:     #F6F7F9;
    --code-text:           #111827;
    --upload-background:   #FFFFFF;
    --upload-file-bg:      #F8F9FA;
    --dot-status:          #15803D;
}

/* System / Browser dark preference (when not explicitly overridden to light) */
@media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]),
    body:not([data-theme="light"]) {
        --app-background:      #0E1117;
        --sidebar-background:  #151922;
        --surface:             #181C24;
        --surface-secondary:   #20242D;
        --surface-hover:       #20242D;
        --border:              #303642;
        --text-primary:        #F1F5F9;
        --text-secondary:      #CBD5E1;
        --text-muted:          #94A3B8;
        --input-background:    #181C24;
        --input-text:          #F1F5F9;
        --input-placeholder:   #94A3B8;
        --accent:              #60A5FA;
        --accent-hover:        #709CF5;
        --code-background:     #161A21;
        --code-text:           #E5E7EB;
        --upload-background:   #181C24;
        --upload-file-bg:      #20242D;
        --dot-status:          #4ADE80;
    }
}

/* Explicit DARK THEME (via Streamlit setting or data-theme) */
[data-theme="dark"],
body[data-theme="dark"],
.stApp[data-theme="dark"] {
    --app-background:      #0E1117;
    --sidebar-background:  #151922;
    --surface:             #181C24;
    --surface-secondary:   #20242D;
    --surface-hover:       #20242D;
    --border:              #303642;
    --text-primary:        #F1F5F9;
    --text-secondary:      #CBD5E1;
    --text-muted:          #94A3B8;
    --input-background:    #181C24;
    --input-text:          #F1F5F9;
    --input-placeholder:   #94A3B8;
    --accent:              #60A5FA;
    --accent-hover:        #709CF5;
    --code-background:     #161A21;
    --code-text:           #E5E7EB;
    --upload-background:   #181C24;
    --upload-file-bg:      #20242D;
    --dot-status:          #4ADE80;
}

/* ═══════════════════════════════════════════════════════════════════════════
   2. CORE APPLICATION CONTAINER
   ═══════════════════════════════════════════════════════════════════════════ */
.stApp {
    background-color: var(--app-background);
    color: var(--text-primary);
    font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
    font-size: 14px;
    line-height: 1.6;
}

.main .block-container {
    max-width: 900px;
    padding-top: 1rem;
    padding-bottom: 5.5rem;
    margin: 0 auto;
    background-color: var(--app-background);
}

/* ═══════════════════════════════════════════════════════════════════════════
   3. HEADER & TOOLBAR (PRESERVE NATIVE SIDEBAR TOGGLE & SETTINGS)
   ═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stHeader"] {
    background-color: var(--app-background);
}

[data-testid="collapsedControl"] {
    color: var(--text-primary);
    display: flex;
    visibility: visible;
    z-index: 100;
}

[data-testid="stSidebarCollapseButton"] {
    color: var(--text-primary);
}

[data-testid="stToolbar"] {
    color: var(--text-secondary);
}

/* ═══════════════════════════════════════════════════════════════════════════
   4. SIDEBAR
   ═══════════════════════════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background-color: var(--sidebar-background);
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] > div:first-child {
    padding: 1.25rem 1rem 1rem;
}

.sb-brand {
    font-size: 15px;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.01em;
    margin-bottom: 0.85rem;
}

.sb-section-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    margin: 0.9rem 0 0.3rem;
}

.sb-status {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12.5px;
    color: var(--text-secondary);
    margin-bottom: 0.75rem;
}

.sb-status .dot-status {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--dot-status);
    flex-shrink: 0;
}

.sb-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 0.75rem 0;
}

.sb-chat-title {
    font-size: 12.5px;
    color: var(--text-secondary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 150px;
    padding: 4px 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
   5. BUTTONS (SIDEBAR & MAIN)
   ═══════════════════════════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    background-color: var(--surface);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    padding: 6px 12px;
    width: 100%;
    box-shadow: none;
    text-align: left;
    transition: background-color 0.15s, border-color 0.15s;
}

section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
    background-color: var(--surface-hover);
    border-color: var(--text-muted);
}

section[data-testid="stSidebar"] div[data-testid="stDownloadButton"] > button {
    background-color: var(--surface);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 12px;
    padding: 4px 8px;
    box-shadow: none;
}

section[data-testid="stSidebar"] div[data-testid="stDownloadButton"] > button:hover {
    background-color: var(--surface-hover);
}

/* Suggestion Buttons in Main Area */
.main div[data-testid="stButton"] > button {
    background-color: var(--surface);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: 8px;
    font-size: 13px;
    padding: 8px 14px;
    box-shadow: none;
    transition: background-color 0.15s, border-color 0.15s;
}

.main div[data-testid="stButton"] > button:hover {
    background-color: var(--surface-hover);
    border-color: var(--accent);
}

/* ═══════════════════════════════════════════════════════════════════════════
   6. INPUTS, TEXTAREAS & SEARCH
   ═══════════════════════════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
.main div[data-testid="stTextInput"] input,
.main div[data-testid="stTextArea"] textarea {
    background-color: var(--input-background);
    color: var(--input-text);
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 13px;
}

section[data-testid="stSidebar"] input::placeholder,
.main div[data-testid="stTextInput"] input::placeholder {
    color: var(--input-placeholder);
}

/* ═══════════════════════════════════════════════════════════════════════════
   7. SELECTBOX (MODEL SELECTOR)
   ═══════════════════════════════════════════════════════════════════════════ */
div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background-color: var(--input-background);
    color: var(--input-text);
    border-color: var(--border);
    border-radius: 6px;
}

div[data-testid="stSelectbox"] svg {
    fill: var(--text-secondary);
}

/* Popover dropdown options */
div[data-baseweb="popover"],
div[data-baseweb="menu"],
ul[role="listbox"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
}

li[role="option"] {
    background-color: var(--surface) !important;
    color: var(--text-primary) !important;
}

li[role="option"]:hover,
li[aria-selected="true"] {
    background-color: var(--surface-hover) !important;
    color: var(--text-primary) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   8. SLIDERS & EXPANDERS
   ═══════════════════════════════════════════════════════════════════════════ */
details[data-testid="stExpander"],
details {
    background-color: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
}

details[data-testid="stExpander"] > summary,
details > summary {
    color: var(--text-primary);
    font-size: 13px;
    font-weight: 500;
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p {
    color: var(--text-secondary);
}

/* Slider track and thumb */
div[data-testid="stSlider"] [role="slider"] {
    background-color: var(--accent);
    border-color: var(--accent);
}

/* ═══════════════════════════════════════════════════════════════════════════
   9. FILE UPLOADER (IMAGE UPLOAD THEME SYNCHRONIZATION)
   ═══════════════════════════════════════════════════════════════════════════ */
div[data-testid="stFileUploader"] {
    background-color: var(--upload-background);
    border: 1px dashed var(--border);
    border-radius: 8px;
    padding: 8px;
}

div[data-testid="stFileUploader"] section {
    background-color: var(--upload-background);
    color: var(--text-primary);
}

div[data-testid="stFileUploader"] section:hover {
    background-color: var(--surface-hover);
}

div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
    background-color: var(--upload-background);
    border-color: var(--border);
}

div[data-testid="stFileUploader"] span,
div[data-testid="stFileUploader"] p,
div[data-testid="stFileUploader"] small {
    color: var(--text-secondary);
}

div[data-testid="stFileUploader"] button {
    background-color: var(--surface);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: 6px;
}

div[data-testid="stFileUploaderFileData"] {
    background-color: var(--upload-file-bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text-primary);
}

.img-preview-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 12px;
    background-color: var(--upload-file-bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text-secondary);
    font-size: 12.5px;
    margin-bottom: 6px;
}

/* ═══════════════════════════════════════════════════════════════════════════
   10. CHAT INPUT
   ═══════════════════════════════════════════════════════════════════════════ */
div[data-testid="stChatInputContainer"] {
    border-top: 1px solid var(--border);
    background-color: var(--app-background);
    padding: 0.6rem 0;
}

div[data-testid="stChatInputContainer"] textarea {
    background-color: var(--input-background);
    color: var(--input-text);
    border: 1px solid var(--border);
    border-radius: 8px;
    font-size: 14px;
}

div[data-testid="stChatInputContainer"] textarea::placeholder {
    color: var(--input-placeholder);
}

div[data-testid="stChatInputContainer"] button {
    color: var(--accent);
}

/* ═══════════════════════════════════════════════════════════════════════════
   11. CHAT MESSAGES (USER RIGHT, AI LEFT)
   ═══════════════════════════════════════════════════════════════════════════ */
div[data-testid="stChatMessage"] {
    background-color: transparent;
    border: none;
    box-shadow: none;
    padding: 0;
}

/* User Message: RIGHT aligned */
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse;
    margin-left: auto;
    margin-right: 0;
    max-width: 75%;
    background-color: var(--surface-secondary);
    border: 1px solid var(--border);
    border-radius: 10px 10px 3px 10px;
    padding: 10px 14px;
}

/* AI Message: LEFT aligned */
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    margin-left: 0;
    margin-right: auto;
    max-width: 90%;
    background-color: transparent;
    border: none;
}

div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] li,
div[data-testid="stChatMessage"] td {
    color: var(--text-primary);
    font-size: 14px;
}

.msg-label-user {
    font-size: 11.5px;
    font-weight: 600;
    color: var(--text-muted);
    text-align: right;
    margin-bottom: 4px;
}

.msg-label-ai {
    font-size: 11.5px;
    font-weight: 600;
    color: var(--text-muted);
    margin-bottom: 4px;
}

/* Action row buttons (Helpful, Regenerate, Copy) */
div[data-testid="column"] div[data-testid="stButton"] > button {
    background-color: transparent;
    border: 1px solid var(--border);
    border-radius: 5px;
    color: var(--text-secondary);
    font-size: 11.5px;
    padding: 3px 9px;
    min-height: 0;
    line-height: 1.4;
    box-shadow: none;
    transition: background-color 0.15s, color 0.15s;
}

div[data-testid="column"] div[data-testid="stButton"] > button:hover {
    background-color: var(--surface-hover);
    color: var(--text-primary);
    border-color: var(--text-muted);
}

/* ═══════════════════════════════════════════════════════════════════════════
   12. MARKDOWN & CODE BLOCKS
   ═══════════════════════════════════════════════════════════════════════════ */
pre,
pre code {
    background-color: var(--code-background) !important;
    color: var(--code-text) !important;
    border: 1px solid var(--border);
    border-radius: 8px;
    font-size: 13px;
}

code:not(pre > code) {
    background-color: var(--code-background);
    color: var(--code-text);
    padding: 1px 5px;
    border-radius: 4px;
    font-size: 12.5px;
}

a {
    color: var(--accent);
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

/* ── App Header ─────────────────────────────────────────────────────────── */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 0 0.75rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.25rem;
}

.app-header-title {
    font-size: 15px;
    font-weight: 600;
    color: var(--text-primary);
}

.app-header-meta {
    font-size: 12px;
    color: var(--text-muted);
}

.app-header-meta code {
    font-size: 11px;
    background-color: var(--surface-secondary);
    border: 1px solid var(--border);
    color: var(--text-secondary);
    border-radius: 4px;
    padding: 1px 5px;
}

/* ── Welcome Screen ─────────────────────────────────────────────────────── */
.welcome-wrap {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    max-width: 560px;
    margin: 1.5rem auto 1.25rem;
}

.welcome-title {
    font-size: 21px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.4rem;
    letter-spacing: -0.02em;
}

.welcome-sub {
    font-size: 13.5px;
    color: var(--text-secondary);
    margin-bottom: 1.5rem;
}

/* ── Responsive ─────────────────────────────────────────────────────────── */
@media (max-width: 768px) {
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        max-width: 90%;
    }
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        max-width: 95%;
    }
    .main .block-container {
        padding: 0 0.75rem 4.5rem;
    }
}
</style>

<script>
(function() {
    function detectAndApplyTheme() {
        const app = document.querySelector('.stApp') || document.body;
        let isDark = false;
        if (app) {
            const bg = window.getComputedStyle(app).backgroundColor;
            const m = bg.match(/\\d+/g);
            if (m && m.length >= 3) {
                // Calculate perceived luminance: 0.299*R + 0.587*G + 0.114*B
                const lum = 0.299 * parseInt(m[0]) + 0.587 * parseInt(m[1]) + 0.114 * parseInt(m[2]);
                isDark = lum < 128;
            } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
                isDark = true;
            }
        } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
            isDark = true;
        }

        const themeStr = isDark ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', themeStr);
        document.body.setAttribute('data-theme', themeStr);
        if (app) {
            app.setAttribute('data-theme', themeStr);
        }
    }

    // Run on startup
    detectAndApplyTheme();
    setTimeout(detectAndApplyTheme, 150);
    setTimeout(detectAndApplyTheme, 500);

    // Observe background color changes on .stApp when user toggles Theme in Streamlit Settings
    try {
        const observer = new MutationObserver(function() {
            detectAndApplyTheme();
        });
        const target = document.querySelector('.stApp') || document.body;
        if (target) {
            observer.observe(target, { attributes: true, attributeFilter: ['class', 'style'] });
        }
    } catch(e) {}

    // Also listen for system theme changes
    if (window.matchMedia) {
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', detectAndApplyTheme);
    }
})();
</script>
"""

def apply_styles() -> None:
    """Inject theme-aware CSS and dynamic theme detection script."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)
