# QnAGPT – Production-Ready Multi-User Agentic AI Workspace

An enterprise-grade, light-mode **Multi-User Agentic AI Platform** powered by **Streamlit**, **LangChain**, **LangGraph**, **Groq**, **Hugging Face Embeddings**, **Qdrant** (`Local` & `Cloud`), **SQLite + SQLAlchemy 2.x + Alembic**, **Local Filesystem Storage**, **Model Context Protocol (MCP)**, and **Deep Web Research**.

---

## 1. Platform Overview

1. **Multi-User Authentication & Strict Isolation**: Email + Password login (`Argon2id` via `argon2-cffi`), Google & GitHub OAuth2 (`Authlib`), server-verified opaque sessions in SQLite, login rate limiting, profile/password management, and permanent account deletion.
2. **Knowledge (Agentic RAG + Qdrant + Local Storage)**: Upload PDF, TXT, DOCX, CSV, and MD documents saved safely inside `storage/users/{user_id}/documents/`, embedded locally with `sentence-transformers/all-MiniLM-L6-v2`, and indexed into **Qdrant** (`QDRANT_MODE=local` or `QDRANT_MODE=cloud`) with mandatory `user_id` & `workspace_id` payload filters. Failed Qdrant uploads retain original files locally for one-click retry.
3. **Connectivity (MCP Host & Client)**: Connect simultaneously to local STDIO and remote Streamable HTTP / SSE MCP servers using the official `mcp` Python SDK (`mcp>=2.3.0`), Official Registry & Glama marketplace discovery, OAuth 2.1 PKCE, and Fernet-encrypted credentials (`mcp_credentials`).
4. **Autonomy & Deep Web Research (LangGraph + HITL)**: Stateful `StateGraph` with Agent vs Response LLM separation (`tool_choice="auto"`), DDGS search + SSRF-protected webpage HTML/table extraction + cross-source verification, and backend-enforced `ApprovalManager` `interrupt()` / `Command(resume=...)`.

---

## 2. Architecture & Design Documentation

Comprehensive engineering documentation lives in [`docs/`](docs/):
- [`docs/HLD.md`](docs/HLD.md) & [`docs/LLD.md`](docs/LLD.md) — High-level & low-level design
- [`docs/DATABASE_SCHEMA.md`](docs/DATABASE_SCHEMA.md) — SQLite + SQLAlchemy 2.x + Alembic ER schema
- [`docs/AUTHENTICATION_FLOW.md`](docs/AUTHENTICATION_FLOW.md) — Argon2id, OAuth2 PKCE, and session flow
- [`docs/AGENT_ARCHITECTURE.md`](docs/AGENT_ARCHITECTURE.md) — LangGraph orchestration & deep web research
- [`docs/MCP_ARCHITECTURE.md`](docs/MCP_ARCHITECTURE.md) — Standalone MCP host & HITL approvals
- [`docs/SECURITY.md`](docs/SECURITY.md), [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md), [`docs/BACKUP_AND_RESTORE.md`](docs/BACKUP_AND_RESTORE.md), [`docs/API_AND_SERVICE_CONTRACTS.md`](docs/API_AND_SERVICE_CONTRACTS.md)

```text
langchain_01/
├── alembic.ini                     # Alembic configuration
├── migrations/                     # Versioned SQLite migrations (0001_initial_platform)
├── docs/                           # Architecture, DB, Auth, Security & Ops docs
├── storage/                        # LocalStorageProvider root (gitignored)
│   ├── backups/                    # Pre-migration & operational snapshots
│   └── users/{user_id}/            # Isolated documents, images, attachments, exports
├── qna_bot/
│   ├── app.py                      # Streamlit entrypoint, auth gate & page dispatcher
│   ├── config/settings.py          # AppSettings & .env loader (Qdrant, OAuth, Groq, MCP)
│   ├── core/                       # SQLite security, Argon2id, rate limiter, exceptions
│   ├── models/                     # SQLAlchemy 2.x ORM models (users, sessions, chats, docs, mcp)
│   ├── schemas/                    # Pydantic v2 DTOs
│   ├── repositories/               # User-scoped transactional SQLite repositories
│   ├── storage/                    # StorageProvider ABC & LocalStorageProvider
│   ├── database/                   # SQLAlchemy engine/WAL PRAGMAs & legacy migration bridge
│   ├── rag/                        # VectorStoreBackend -> QdrantVectorStoreManager (+ FAISS bridge)
│   ├── mcp/                        # Official MCP SDK host, vaults, OAuth PKCE & marketplaces
│   ├── tools/                      # Multi-backend web search, webpage reader & RAG tools
│   ├── agent/                      # LangGraph StateGraph, memory checkpointer, HITL, mcp_adapter
│   ├── services/                   # Auth, User, Document, Conversation, Chat, MCP, Suggestion
│   └── ui/                         # Light-mode SaaS shell, Auth & Account pages, workspace views
└── tests/                          # Automated unit, integration & Streamlit AppTest suites
```

---

## 3. Setup & Running Locally

1. **Install dependencies**:
   ```powershell
   uv sync
   ```
2. **Start Qdrant locally (optional if using Qdrant Cloud)**:
   ```powershell
   docker run -d --name qnagpt-qdrant -p 6333:6333 -p 6334:6334 -v "${PWD}\data\qdrant_storage:/qdrant/storage" qdrant/qdrant:latest
   ```
3. **Configure `.env`** (see [`.env.example`](.env.example)):
   - Required for LLM chat: `GROQ_API_KEY`
   - Optional OAuth2: `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET`
   - Vector DB: `QDRANT_MODE=local` (`QDRANT_URL=http://localhost:6333`) or `QDRANT_MODE=cloud` (`QDRANT_URL`, `QDRANT_API_KEY`)
4. **Apply database migrations**:
   ```powershell
   .\.venv\Scripts\alembic.exe upgrade head
   ```
5. **Run the application**:
   ```powershell
   .\.venv\Scripts\streamlit.exe run qna_bot/app.py
   ```

---

## 4. Automated Verification

```powershell
.\.venv\Scripts\pytest.exe -v
```
