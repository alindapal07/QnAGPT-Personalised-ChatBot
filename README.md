# Enterprise AI Assistant (`qna_bot`)

A modular, production-structured multimodal AI Q&A chatbot built with **Streamlit**, **LangChain**, and **Groq**.

---

## Project Structure

```text
langchain_01/
├── qna_bot/                     # Production application package
│   ├── __init__.py              # Package metadata & public exports
│   ├── __main__.py              # `python -m qna_bot` entry point
│   ├── app.py                   # Streamlit orchestrator entry point
│   ├── cli.py                   # CLI launcher (`qna-bot`)
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py          # Immutable AppSettings & .env loading
│   ├── core/
│   │   ├── __init__.py
│   │   ├── exceptions.py        # Domain-specific exception hierarchy
│   │   └── logging_config.py    # Idempotent structured logging setup
│   ├── services/
│   │   ├── __init__.py
│   │   ├── agent_service.py     # LangChain + ChatGroq routing & streaming
│   │   ├── export_service.py    # JSON, Markdown, and TXT export serializers
│   │   ├── image_service.py     # Image validation, MIME detection & base64 encoding
│   │   └── session_service.py   # Session state lifecycle & recent chat storage
│   └── ui/
│       ├── __init__.py
│       ├── assets/
│       │   ├── styles.css       # Light, Dark & Auto semantic CSS variables
│       │   └── theme.js         # Dynamic theme attribute synchronization
│       ├── chat_view.py         # Chat history & message action controls
│       ├── components.py        # Header, welcome screen, copy button & image uploader
│       ├── sidebar.py           # Navigation, search, recent chats, settings & export
│       └── styles.py            # Cached stylesheet/script asset injector
│
├── tests/                       # Automated unit & Streamlit AppTest suite
├── archive/                     # Non-production, experimental & legacy files
├── .env.example                 # Environment variable template
├── pyproject.toml               # Package metadata & dependencies
├── requirements.txt             # Pip-compatible dependency list
└── uv.lock                      # Locked dependency graph
```

---

## Production Coding Flow

The codebase separates concerns into four strict layers so UI, business logic, configuration, and infrastructure remain decoupled and testable:

1. **Configuration (`qna_bot/config/`)**: Loads `.env` safely and provides a typed, immutable `AppSettings` object (`get_settings()`).
2. **Core Infrastructure (`qna_bot/core/`)**: Provides domain exceptions (`ConfigurationError`, `ImageValidationError`, `AgentExecutionError`) and idempotent logging (`configure_logging()`).
3. **Services (`qna_bot/services/`)**: Pure business logic with no Streamlit rendering dependencies (except `session_service` abstracting `st.session_state`):
   - `agent_service.py`: Converts conversation history into LangChain messages (`SystemMessage`, `HumanMessage`, `AIMessage`), routes text queries to the selected text model and image queries to the vision model, and streams response chunks.
   - `image_service.py`: Validates uploaded image size (`<= 10 MB`), verifies format/integrity via Pillow, and returns base64 payloads.
   - `export_service.py`: Serializes chat history to JSON, Markdown, and plain text.
   - `session_service.py`: Manages session initialization, new/clear chat transitions, and recent conversation snapshots.
4. **Presentation (`qna_bot/ui/`) & Orchestrator (`qna_bot/app.py`)**: `app.py` orchestrates page config, styling, sidebar controls, user submissions, and AI streaming turns using modular UI components.

---

## Models & Multimodal Handling

- **Text Models** (selectable in sidebar): `openai/gpt-oss-20b` (default), `openai/gpt-oss-120b`, `qwen/qwen3.8-27b`
- **Vision Model** (auto-routed when an image is attached): `qwen/qwen3.8-27b`
- **Supported Image Formats**: `png`, `jpg`, `jpeg`, `webp` (up to 10 MB)

---

## Setup & Installation

Using `uv` (recommended):

```bash
uv sync
```

Or using `pip`:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the repository root (see `.env.example`):

```env
GROQ_API_KEY=your_groq_api_key_here
LOG_LEVEL=INFO
```

---

## Running the Application

From the repository root:

```bash
uv run streamlit run qna_bot/app.py
```

Or using the installed CLI / module entrypoint:

```bash
uv run qna-bot
# or
python -m qna_bot
```

Or from inside `qna_bot/`:

```bash
streamlit run app.py
```

---

## Running Tests

```bash
python -m unittest discover -s tests -v
```
