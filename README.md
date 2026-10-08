# Enterprise AI Assistant

A clean, modular, production-grade AI chat application built with Streamlit, LangChain, and Groq.

---

## Architecture

The project is structured into clean, decoupled modules:

```
qna_bot/
│
├── app.py          # Application entry point & orchestration
├── agent.py        # LangChain & ChatGroq AI logic (text & multimodal)
├── styles.py       # Theme-aware CSS (Light, Dark, and Auto)
├── ui.py           # Streamlit UI presentation & layout components
├── config.py       # Configuration & environment variables
└── requirement.txt # Project dependencies
```

- **`app.py`**: Lightweight orchestrator managing session state, theme loading, and calling UI/agent workflows.
- **`agent.py`**: Encapsulates `ChatGroq`, multimodal message construction, streaming responses, and routing.
- **`styles.py`**: Pure CSS variables (`--app-bg`, `--surface`, `--border`, `--upload-bg`, etc.) supporting Streamlit's Light, Dark, and Auto themes. Preserves native sidebar collapse/reopen toggle.
- **`ui.py`**: Pure Streamlit rendering functions for header, sidebar, chat rows (User on RIGHT, AI on LEFT), image previews, and JS-based copy actions.
- **`config.py`**: Centralized constants, model identifiers, token limits, and environment variable loading.

---

## Models & Multimodal Handling

- **Text Model**: `openai/gpt-oss-20b` (Default, selectable in sidebar: `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `qwen/qwen3.8-27b`)
- **Vision Model**: `qwen/qwen3.8-27b` (Automatically routed when an image is attached)
- **Automatic Routing**: Normal text questions are handled by the selected text model. Whenever an image is uploaded (PNG, JPG, JPEG, WEBP), requests are routed to the vision model with full streaming responses.

---

## Installation

```bash
pip install -r requirement.txt
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## Running the Application

From the project root:

```bash
streamlit run langchain_01/qna_bot/app.py
```

Or from within `langchain_01/qna_bot/`:

```bash
streamlit run app.py
```

---

## Key Features

1. **Enterprise Theme Support**: Automatically respects Streamlit's Light, Dark, and Auto themes. All elements—including the image uploader, dropzone, text inputs, and buttons—adapt seamlessly with zero contrast issues.
2. **Native Sidebar Controls**: Fully collapsible sidebar that preserves Streamlit's native reopen toggle (`collapsedControl`).
3. **ChatGPT-Style Layout**: User messages are aligned to the **RIGHT**; AI messages are aligned to the **LEFT**.
4. **Multimodal Image Support**: Attach PNG, JPG, JPEG, and WEBP images with live previews, removal controls, and inline message rendering.
5. **Real Streaming & Action Controls**: Smooth token streaming with inline copy (copies to clipboard and displays "Copied" for 1.8s without opening modals), regenerate, and feedback controls.

