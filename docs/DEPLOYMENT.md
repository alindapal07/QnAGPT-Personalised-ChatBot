# QnAGPT Deployment & Operations Guide

## 1. Supported Runtime Environment

- **OS**: Windows / Linux / macOS
- **Python**: `3.12+` (`.venv` recommended via `uv` or `pip`)
- **Frontend & Server**: Streamlit `>=1.65.0`
- **Primary Relational DB**: SQLite (zero external SQL daemon required; uses `WAL` mode + `PRAGMA foreign_keys = ON`)
- **Vector Database**: Qdrant (`QDRANT_MODE=local` via Docker/local service OR `QDRANT_MODE=cloud` via Qdrant Cloud)
- **File Storage**: Local filesystem (`./storage` — no AWS S3 required)

---

## 2. Environment Variables (`.env`)

Configure in `.env` (or `data/.env`):

```ini
# --- Core LLM & Application ---
GROQ_API_KEY=gsk_...
LOG_LEVEL=INFO
QNAGPT_DATA_DIR=./data
QNAGPT_STORAGE_DIR=./storage

# --- Database & Migrations ---
# Defaults cleanly to sqlite:///./data/chatbot.db
DATABASE_URL=sqlite:///./data/chatbot.db
LANGGRAPH_CHECKPOINT_DB=./data/checkpoints.db

# --- Security & Authentication ---
# Generate 32-byte URL-safe Fernet key / high-entropy secret:
# python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
QNAGPT_CREDENTIAL_KEY=
SESSION_SECRET_KEY=
SESSION_TTL_HOURS=168
APP_BASE_URL=http://localhost:8501

# --- Optional OAuth2 Login (Google & GitHub via Authlib) ---
# When unset, Email/Password auth works normally and OAuth buttons display configuration guidance.
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GITHUB_CLIENT_ID=
GITHUB_CLIENT_SECRET=

# --- Vector Database (Qdrant) ---
QDRANT_MODE=local
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=qna_documents

# --- MCP Marketplaces & Timeouts ---
MCP_REGISTRY_URL=https://registry.modelcontextprotocol.io
GLAMA_API_URL=https://glama.ai/api/mcp/v1
GLAMA_API_KEY=
QNAGPT_REGISTRY_CACHE_TTL_SEC=300
QNAGPT_OAUTH_REDIRECT_URI=http://localhost:8501/oauth/callback
```

---

## 3. Running Qdrant (Local Docker vs Qdrant Cloud)

### Option A: Local Qdrant via Docker Desktop
```powershell
docker run -d --name qnagpt-qdrant `
  -p 6333:6333 -p 6334:6334 `
  -v "${PWD}\data\qdrant_storage:/qdrant/storage" `
  qdrant/qdrant:latest
```
Verify readiness: `Invoke-WebRequest http://localhost:6333/readyz`.
Set `QDRANT_MODE=local`, `QDRANT_URL=http://localhost:6333`, and leave `QDRANT_API_KEY=` empty.

### Option B: Qdrant Cloud
1. Create a Qdrant Cloud cluster (`https://<cluster-id>.<region>.aws.cloud.qdrant.io:6333`) and API key.
2. Set `QDRANT_MODE=cloud`, `QDRANT_URL=https://...`, and `QDRANT_API_KEY=...` in `.env`.
3. Embedding dimension for `sentence-transformers/all-MiniLM-L6-v2` is `384` (Cosine similarity). `QdrantVectorStoreManager` ensures payload indexes on `user_id`, `workspace_id`, and `document_id`.

---

## 4. Database Initialization & Migrations

1. Apply Alembic schema migrations before launching in production:
   ```powershell
   .\.venv\Scripts\alembic.exe upgrade head
   ```
2. Run the legacy bridge check (if upgrading an existing `./data/chatbot.db` or `./data/vector_store/local_user/` FAISS index) cleanly so pre-existing threads, `NeuralVerge` MCP credentials, and indexed PDFs transfer without data loss.

---

## 5. Launching & Verifying the Application

1. **Install dependencies**:
   ```powershell
   pip install -e .
   ```
2. **Pre-run automated tests**:
   ```powershell
   .\.venv\Scripts\pytest.exe -v
   ```
3. **Start Streamlit Server**:
   ```powershell
   .\.venv\Scripts\streamlit.exe run qna_bot/app.py --server.port 8501 --server.address 127.0.0.1
   ```
4. **Reverse Proxy & Production Hardening**:
   - Terminate TLS (`https://`) at Nginx / Caddy / cloud load balancer before forwarding WebSocket upgrades (`/ _stcore/stream`) to `127.0.0.1:8501`.
   - Mount `./data` and `./storage` on persistent encrypted storage volumes (`chmod 700` / restrictive Windows ACLs).
   - Never run `debugpy` or expose `--server.enableCORS=false` publicly.
