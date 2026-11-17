# QnAGPT Production Troubleshooting Guide

**Project**: QnAGPT – Production-Ready Multi-User Agentic AI Workspace  
**Status**: Verified Operational Guide  
**Author**: Principal Software Architect & Senior Python Backend Engineer  

---

## 1. Common Operational Issues & Root-Cause Solutions

### 1. `OperationalError: (sqlite3.OperationalError) no such column: is_pinned`
- **Symptom**: Application startup crashes with `OperationalError: no such column: is_pinned` during thread retrieval.
- **Root Cause**: The SQLite database (`data/chatbot.db`) was initialized before column additions in newer platform revisions.
- **Resolution**:
  - The application includes an automated dynamic schema synchronization bridge in `qna_bot/database/legacy_migration.py` (`ensure_legacy_workspace_synced`), which automatically runs `ALTER TABLE threads ADD COLUMN is_pinned INTEGER DEFAULT 0` on startup.
  - To apply migrations manually:
    ```powershell
    uv run alembic upgrade head
    ```

---

### 2. `groq.APIConnectionError: Connection error.`
- **Symptom**: Chat responses fail with `Unable to contact the model: Connection error.`
- **Root Cause**:
  1. `GROQ_API_KEY` is missing, malformed, or has expired quotas.
  2. The application is running inside a restrictive sandbox or firewall blocking outbound HTTPS requests to `api.groq.com:443`.
- **Resolution**:
  - Verify your `.env` contains a valid key: `GROQ_API_KEY=gsk_...`.
  - In unit tests, ensure tests mock `ChatGroq` via `unittest.mock.patch("qna_bot.agent.nodes.ChatGroq", return_value=fake_llm)`.

---

### 3. `ModuleNotFoundError: No module named 'mcp'`
- **Symptom**: Python raises `ModuleNotFoundError` when loading MCP client modules.
- **Root Cause**: Python's import resolver shadowed the official PyPI `mcp` SDK with the local internal directory `qna_bot/mcp/`.
- **Resolution**:
  - In architecture v2.0.0, the internal package has been permanently renamed to `qna_bot/mcp_client/`.
  - Update any custom import statements to `from qna_bot.mcp_client import ...`.
  - Ensure `qna_bot/__init__.py`'s `sanitize_import_path()` is executed before external runners load the package.

---

### 4. `sqlite3.OperationalError: database is locked`
- **Symptom**: Simultaneous user operations or background MCP audit logs fail with `database is locked`.
- **Root Cause**: Concurrency contention on SQLite when running multiple concurrent writers without Write-Ahead Logging (WAL) enabled.
- **Resolution**:
  - Verify WAL mode and 10-second busy timeout are active:
    ```sql
    PRAGMA journal_mode = WAL;
    PRAGMA busy_timeout = 10000;
    ```
  - Both `SQLAlchemyDatabase` and `DatabaseManager` apply these PRAGMAs automatically on every connection check-out.

---

### 5. Qdrant Connection Refused (`http://localhost:6333`)
- **Symptom**: Knowledge Base displays `Qdrant (LOCAL) is unreachable: Connection refused`.
- **Root Cause**: The local Docker container is stopped or has not been created.
- **Resolution**:
  - Start the local container:
    ```powershell
    docker start qnagpt-qdrant
    # or create a new one:
    docker run -d --name qnagpt-qdrant -p 6333:6333 -p 6334:6334 -v "${PWD}\data\qdrant_storage:/qdrant/storage" qdrant/qdrant:latest
    ```
  - **Graceful Degradation**: Files uploaded while Qdrant is offline are safely preserved in `storage/users/{user_id}/documents/`. Once Qdrant is online, navigate to **Knowledge Base** and click **Retry Indexing** to index preserved documents with one click.
  - To use cloud-hosted Qdrant, update `.env`:
    ```ini
    QDRANT_MODE=cloud
    QDRANT_URL=https://your-cluster-id.qdrant.tech:6333
    QDRANT_API_KEY=your_qdrant_cloud_api_key
    ```

---

### 6. `tool_use_failed` or Unexpected Harmony Tool Call Loops
- **Symptom**: LLM outputs raw JSON tool schemas or fails with `tool_choice is none` error.
- **Root Cause**: Some Groq/OpenAI fine-tuned models hallucinate tool calls when synthesizing answers if tools are bound to the final response prompt.
- **Resolution**:
  - The architecture enforces strict **Agent vs. Response LLM separation**:
    - `_make_agent_llm()` binds tools with `tool_choice="auto"` during planning and execution.
    - `_make_response_llm()` is **never** bound to tools, and `_messages_for_response_llm()` flattens previous tool traces into plain text evidence blocks.
  - If a model still emits tool calls unexpectedly, `_invoke_response_llm()` automatically catches `_is_tool_use_failed_error(exc)` and retries with a strict Markdown-only synthesis prompt.

---

### 7. Human-in-the-Loop (HITL) Action Blocked / Rejected
- **Symptom**: An agent action fails with `Execution refused: valid backend approval matching exact arguments was not granted`.
- **Root Cause**: The tool requires human approval (`MODIFICATION` or `SENSITIVE`), and either:
  1. The user clicked "Reject Action" in the UI.
  2. The tool parameters were modified between the approval creation and consumption (preventing argument-tampering attacks).
- **Resolution**:
  - Review the tool parameters in the **Agent Workspace** approval modal.
  - If you wish to allow the tool to execute automatically without approval prompts in the future, navigate to **Tool Marketplace**, locate the tool, and check **Auto-approve**.
