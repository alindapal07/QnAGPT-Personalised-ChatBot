# QnAGPT Database Schema (SQLite + SQLAlchemy 2.x + Alembic)

## 1. Engine Configuration

- Foreign keys enforced on every SQLite connection: `PRAGMA foreign_keys = ON`.
- Concurrency & durability: `PRAGMA journal_mode = WAL`, `PRAGMA busy_timeout = 10000`, `PRAGMA synchronous = NORMAL`.
- Managed via Alembic migrations in `migrations/`.

## 2. Entity Relationship Diagram

```mermaid
erDiagram
    users ||--o{ auth_identities : links
    users ||--o{ user_sessions : owns
    users ||--o{ conversations : owns
    users ||--o{ uploaded_documents : uploads
    users ||--o{ mcp_servers : configures
    users ||--o{ user_settings : stores
    users ||--o{ audit_logs : generates
    users ||--o{ feedback_records : submits

    conversations ||--o{ messages : contains
    conversations ||--o{ agent_executions : tracks
    conversations ||--o{ mcp_execution_logs : records
    mcp_servers ||--o{ mcp_execution_logs : produces
    mcp_servers ||--o{ mcp_tool_permissions : governs
    mcp_servers ||--o{ mcp_credentials : secures
    messages ||--o{ feedback_records : rates

    users {
        string id PK
        string email UK
        string password_hash
        string full_name
        string avatar_url
        boolean is_active
        boolean is_verified
        datetime created_at
        datetime updated_at
        datetime last_login_at
    }

    auth_identities {
        string id PK
        string user_id FK
        string provider
        string provider_user_id
        string provider_email
        datetime created_at
    }

    user_sessions {
        string id PK
        string user_id FK
        string token_hash UK
        string csrf_token
        datetime expires_at
        datetime revoked_at
        datetime created_at
    }

    conversations {
        string id PK
        string user_id FK
        string title
        string description
        string model_name
        string agent_mode
        string status
        boolean is_pinned
        datetime created_at
        datetime updated_at
        datetime archived_at
    }

    messages {
        int id PK
        string conversation_id FK
        string user_id FK
        string role
        text content
        string message_type
        int token_usage
        text metadata_json
        datetime created_at
    }

    uploaded_documents {
        string id PK
        string user_id FK
        string workspace_id
        string filename
        string storage_path
        string file_type
        int file_size
        string checksum
        string processing_status
        int chunk_count
        int page_count
        string qdrant_collection
        text error_message
        datetime created_at
        datetime updated_at
    }

    mcp_servers {
        string id PK
        string user_id FK
        string server_name
        string transport_type
        string endpoint
        text configuration_json
        string authentication_type
        boolean enabled
        string connection_status
        datetime created_at
        datetime updated_at
    }

    agent_executions {
        string id PK
        string user_id FK
        string conversation_id FK
        string execution_id UK
        string status
        string current_step
        text execution_metadata
        datetime started_at
        datetime completed_at
    }

    mcp_execution_logs {
        string id PK
        string user_id FK
        string conversation_id FK
        string server_id FK
        string tool_name
        string status
        float execution_time
        text error_message
        datetime created_at
    }

    user_settings {
        int id PK
        string user_id FK
        string setting_key
        text setting_value
        datetime updated_at
    }

    audit_logs {
        int id PK
        string user_id FK
        string action
        string resource_type
        string resource_id
        text metadata_json
        datetime created_at
    }

    feedback_records {
        string id PK
        string user_id FK
        int message_id FK
        string rating
        text comment
        datetime created_at
    }
```

## 3. Core Indexes & Constraints

- `users`: unique index on lower(`email`).
- `auth_identities`: unique composite constraint `(provider, provider_user_id)` and index on `user_id`.
- `user_sessions`: unique index on `token_hash`, index on `(user_id, expires_at)`.
- `conversations`: composite index `(user_id, status, is_pinned, updated_at)`.
- `messages`: composite index `(conversation_id, user_id, created_at)`.
- `uploaded_documents`: composite index `(user_id, checksum)` for fast per-user deduplication + `(user_id, workspace_id, processing_status)`.
- `mcp_servers`: unique `(user_id, server_name)` + index `(user_id, enabled)`.
- `mcp_tool_permissions`: PK / unique `(user_id, server_id, tool_name)`.
- `mcp_credentials`: unique `(user_id, server_id, auth_type)` storing only Fernet ciphertexts.
- `user_settings`: unique `(user_id, setting_key)`.
- `audit_logs`, `agent_executions`, `mcp_execution_logs`: indexes on `(user_id, created_at DESC)` and `conversation_id`.

## 4. Legacy Migration Bridge (`chatbot.db` -> Normalized Schema)

Existing `data/chatbot.db` data (`threads`, `thread_messages`, `mcp_servers`, `mcp_credentials`, `mcp_tool_permissions`, `mcp_approvals`, `mcp_audit_logs`) belongs to legacy workspace `local_user`. During Alembic + migration execution:
- Legacy tables stay intact/backed up first (`storage/backups/`).
- A seeded legacy/local workspace user record receives existing threads, messages, MCP registrations, and encrypted credentials cleanly without changing `(user_id, server_id)` vault keys or breaking LangGraph checkpoints.
