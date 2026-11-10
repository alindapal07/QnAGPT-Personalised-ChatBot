# QnAGPT – Personalized Agentic RAG Assistant

An enterprise-grade, multimodal **Agentic RAG Chatbot** built with **Streamlit**, **LangChain**, **LangGraph**, **Groq**, **Hugging Face Embeddings**, **FAISS**, **DuckDuckGo**, and **SQLite**.

---

## Architecture Overview

The application is divided into two cooperating subsystems orchestrated by a compiled **LangGraph `StateGraph`** inside [`qna_bot/`](qna_bot/):

1. **Document Intelligence (`qna_bot/rag/`)**:
   - Multi-format ingestion for **PDF**, **TXT**, **DOCX**, and **CSV** (`PyPDFLoader`, `TextLoader`, `Docx2txtLoader`, `CSVLoader`).
   - Size/extension validation, temporary file isolation, text cleaning, recursive chunking (`CHUNK_SIZE = 1000`, `CHUNK_OVERLAP = 200`), and SHA-256 deduplication.
   - Local **Hugging Face** embeddings (`sentence-transformers/all-MiniLM-L6-v2`, CPU/CUDA, normalized vectors, no OpenAI key required).
   - User-scoped **FAISS** persistence under `data/vector_store/<user_id>/` with SHA-256 index integrity verification and a modular `VectorStoreBackend` interface ready for future Qdrant Cloud migration.
2. **Agentic Intelligence (`qna_bot/agent/` & `qna_bot/tools/`)**:
   - Stateful **LangGraph `StateGraph`** with conditional routing and bounded retries:
     `analyze_query` -> (`direct_answer` + `ToolNode` | `rewrite_query` -> `retrieve_documents` -> `grade_documents` | `web_search`) -> `generate_answer` -> `validate_answer`.
   - **Query Analysis & Rewriting**: Classifies queries (`GENERAL`, `DOCUMENT`, `WEB_SEARCH`, `FOLLOW_UP`) and turns conversational follow-ups into standalone search queries.
   - **Document Grading & Validation**: Filters irrelevant chunks, prevents infinite loops (`max_retrieval_retries = 2`, `max_validation_retries = 1`), and validates document-grounded citations.
   - **Web Search & Tools**: DuckDuckGo search preserving source URLs alongside user-scoped document tools (`retrieve_uploaded_documents`, `summarize_uploaded_documents`).
   - **Persistent SQLite Memory**: `langgraph-checkpoint-sqlite` (`SqliteSaver`) in `data/chatbot.db` with unique `thread_id`s and server-side `user_id` isolation.

---

## Project Structure

```text
langchain_01/
├── qna_bot/
│   ├── __init__.py
│   ├── __main__.py
│   ├── app.py                      # Streamlit entrypoint & workflow orchestrator
│   ├── cli.py                      # CLI runner (`qna-bot`)
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py             # Immutable AppSettings, .env & Streamlit secrets support
│   ├── core/
│   │   ├── __init__.py
│   │   ├── exceptions.py           # Typed domain exceptions
│   │   └── logging_config.py       # Idempotent structured logger
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── graph.py                # Compiled LangGraph StateGraph + SqliteSaver
│   │   ├── memory.py               # Thread management & SQLite checkpointing
│   │   ├── nodes.py                # All 8 graph nodes + ToolNode wiring
│   │   ├── prompts.py              # Classification, rewrite, RAG & validation prompts
│   │   ├── routing.py              # Deterministic conditional edge routers
│   │   └── state.py                # Typed AgentState with add_messages
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── chunking.py             # Text cleaning & RecursiveCharacterTextSplitter
│   │   ├── embeddings.py           # Lazy singleton HuggingFaceEmbeddings
│   │   ├── grader.py               # Relevance grading (RELEVANT/PARTIALLY_RELEVANT/IRRELEVANT)
│   │   ├── ingestion.py            # Validation, deduplication & batch indexing
│   │   ├── loaders.py              # PDF, TXT, DOCX, CSV document loaders
│   │   ├── retriever.py            # User-scoped top-k & MMR retriever
│   │   └── vector_store.py         # VectorStoreBackend & verified FAISS persistence
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── document_tools.py       # User-isolated LangChain document tools
│   │   └── web_search.py           # DuckDuckGo web search tool & URL preservation
│   ├── services/
│   │   ├── __init__.py
│   │   ├── agent_service.py        # ChatGroq streaming & multimodal vision fallback
│   │   ├── chat_service.py         # LangGraph turn execution & thread persistence
│   │   ├── document_service.py     # Knowledge base upload, listing & removal
│   │   ├── export_service.py       # JSON, Markdown & TXT chat exporters
│   │   ├── image_service.py        # Image validation & base64 encoding
│   │   └── session_service.py      # Streamlit session state lifecycle
│   └── ui/
│       ├── __init__.py
│       ├── assets/
│       │   ├── styles.css          # Light, Dark & Auto semantic CSS
│       │   └── theme.js            # Dynamic theme synchronization
│       ├── chat_view.py            # Chat bubbles, citations & action controls
│       ├── components.py           # Header, welcome, sources panel, copy & image UI
│       ├── sidebar.py              # Mode selector, KB manager, threads, settings, export
│       └── styles.py               # Cached asset injector
├── tests/                          # Pytest test suite
├── archive/                        # Reference / non-production legacy files
├── .env.example                    # Environment template
├── pyproject.toml                  # Dependencies & pytest/uv config
├── requirements.txt                # Pip requirements (Streamlit Cloud compatible)
└── uv.lock                         # Locked uv dependency graph
```

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

Create a `.env` file in the repository root (see [`.env.example`](.env.example)):

```env
GROQ_API_KEY=your_groq_api_key_here
LOG_LEVEL=INFO
```

*No `OPENAI_API_KEY` is required. Embeddings run locally via `sentence-transformers/all-MiniLM-L6-v2`.*

---

## Running Locally

```bash
uv run streamlit run qna_bot/app.py
```

Alternative entrypoints:

```bash
uv run qna-bot
# or
python -m qna_bot
```

---

## Chat Modes & Features

- **Auto (Agentic)**: Classifies intent, routes between general conversation, RAG over uploaded documents, tool calling, and DuckDuckGo web search automatically.
- **Document Chat**: Prioritizes retrieval, semantic grading, citation validation, and expandable document source previews (filename + page number).
- **General Chat**: Direct Groq conversation and existing multimodal vision analysis (`PNG`, `JPG`, `JPEG`, `WEBP`).
- **Workspace Isolation**: Documents and SQLite conversation threads are partitioned by `user_id` (`Workspace ID`) on the server side.

---

## Testing

```bash
pytest -v
```

---

## Streamlit Community Cloud Deployment

1. Push the repository (without `.env`) to GitHub.
2. In Streamlit Community Cloud, set the **Main file path** to `qna_bot/app.py`.
3. Add `GROQ_API_KEY` under **App Settings -> Secrets** (`settings.py` reads both environment variables and `st.secrets`).
4. Dependencies are picked up automatically from `requirements.txt` / `pyproject.toml`.

> **Persistence Note:** Local FAISS indexes (`data/vector_store/`) and SQLite files (`data/chatbot.db`) are suitable for local/single-instance usage, whereas ephemeral cloud containers reset local disk on restart. `VectorStoreBackend` in `qna_bot/rag/vector_store.py` is structured so you can swap in **Qdrant Cloud** and managed PostgreSQL for multi-user cloud persistence without rewriting the RAG or LangGraph pipeline.
