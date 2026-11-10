# QnAGPT Backup and Restore Runbook

## 1. Critical Assets Requiring Coordinated Backup

| Asset | Default Path | Why It Is Critical |
|---|---|---|
| Primary SQLite Database | `data/chatbot.db` | Stores users, OAuth identities, sessions, conversations, messages, document metadata, MCP configs, audit logs, and encrypted secrets. |
| LangGraph Checkpoints DB | `data/checkpoints.db` (or `chatbot.db` checkpointer tables) | Stores resumable graph thread states and pending HITL interrupts. |
| Vault Encryption Key | `data/.mcp_vault.key` (or `QNAGPT_CREDENTIAL_KEY` env secret) | **Without this key, encrypted `mcp_credentials` rows cannot be decrypted.** Store in a separate secrets manager. |
| Local File Storage | `storage/users/{user_id}/` | Holds original uploaded PDFs/DOCX/CSV/MD/TXT files, chat image attachments, and exports. |
| Qdrant Vector Store | Local `./data/qdrant_storage` volume OR Qdrant Cloud snapshots | Stores `384`-dim embeddings and chunk metadata (can also be rebuilt cleanly from `storage/users/` raw files!). |

---

## 2. Backup Procedure (Zero-Corruption Online SQLite + Files + Qdrant)

1. **Quiesce or Perform Online SQLite Backup**:
   - Never copy raw `.db` + `-wal` files mid-write with `cp`/`Copy-Item` while the app runs—use SQLite's online backup API (`sqlite3.Connection.backup`) after creating `storage/backups/{timestamp}/`:
     ```python
     import sqlite3
     src = sqlite3.connect("data/chatbot.db")
     dst = sqlite3.connect("storage/backups/2026-11-10/chatbot.db")
     with dst:
         src.backup(dst)
     dst.close()
     src.close()
     ```
2. **Archive Local Uploaded Files**:
   - Compress `./storage/users/` into `storage/backups/{timestamp}/users_files.zip` (exclude `storage/backups/` to prevent recursive archives).
3. **Create Qdrant Snapshot**:
   - Via `qdrant-client`: `client.create_snapshot(collection_name="qna_documents")` and download/retain the `.snapshot` file, **OR** back up `./data/qdrant_storage/` while the local Qdrant container is stopped.
4. **Verify Checksums**:
   - Record `SHA-256` digests of the backed-up `.db`, file archive, and Qdrant snapshot. Store `.mcp_vault.key` encrypted separately from database dumps.

---

## 3. Restore Procedure

1. **Stop the Running Streamlit Server** (and local Qdrant container if replacing raw container storage).
2. **Restore Vault Key / Env Secrets**:
   - Place `QNAGPT_CREDENTIAL_KEY` (or `data/.mcp_vault.key`) before starting any database or MCP service.
3. **Restore SQLite Database & Verify Integrity**:
   - Move current `data/chatbot.db*` (`-wal`, `-shm`) to a safety folder.
   - Copy verified backup `chatbot.db` to `data/chatbot.db` and run:
     ```powershell
     .\.venv\Scripts\python.exe -c "import sqlite3; c=sqlite3.connect('data/chatbot.db'); print(c.execute('PRAGMA integrity_check;').fetchall()); print(c.execute('PRAGMA foreign_key_check;').fetchall())"
     ```
   - Run `.\.venv\Scripts\alembic.exe upgrade head` if restoring onto a newer application release.
4. **Restore Local Storage**:
   - Extract `users_files.zip` back into `./storage/users/`, confirming directory ownership matches uploaded document paths (`uploaded_documents.storage_path`).
5. **Restore Qdrant Vectors (Two Options)**:
   - **Fast Recover**: Upload/recover the `.snapshot` into Qdrant (`client.recover_snapshot(...)` or REST `/collections/qna_documents/snapshots/recover`).
   - **Full Rebuild**: Even if Qdrant snapshots are lost, invoke the document re-index pipeline over all `uploaded_documents` whose original files are safely preserved in `storage/users/{user_id}/documents/`.

---

## 4. Legacy Data Preservation Note (`chatbot.db` & `Payslip_9_2026.pdf`)

Before running destructive Phase 2–6 migrations against the existing workspace `data/`:
1. Snapshot `data/chatbot.db`, `data/.mcp_vault.key`, and `data/vector_store/local_user/`.
2. Pre-existing `17` threads, `22` messages, `NeuralVerge` MCP registration/credentials, and FAISS document texts (`Payslip_9_2026.pdf`) are preserved/migrated without breaking Fernet keys or checkpoints.
