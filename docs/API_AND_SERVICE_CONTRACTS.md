# QnAGPT API & Service Contracts

All backend services act as the authoritative trust boundary: every user-scoped method accepts a backend-verified `user_id` (resolved from `AuthService` / active `UserSession`, never trusted directly from frontend widgets) and enforces ownership checks inside database transactions.

---

## 1. Authentication & User Services (`qna_bot/services/`)

### `AuthService` (`qna_bot/services/auth_service.py`)
- `register_user(email: str, password: str, full_name: str = "") -> AuthResult`
  - Normalizes email, checks uniqueness, enforces password complexity (`>=8` chars, upper/lower/digit-or-symbol), hashes via `Argon2id`, assigns UUID `user.id`, creates default `user_settings` and initial session, writes `audit_logs`.
- `login_with_password(email: str, password: str, client_key: str = "local") -> AuthResult`
  - Enforces sliding-window rate limits (`5 attempts / 5 min`), verifies Argon2id hash in constant time, updates `last_login_at`, returns opaque signed session token + `UserProfile`.
- `start_oauth_login(provider: Literal["google", "github"]) -> OAuthRedirect`
  - Validates provider configuration (`GOOGLE_CLIENT_ID`/`SECRET` or `GITHUB_CLIENT_ID`/`SECRET`; raises `ConfigurationError` if unconfigured—never mocks login), persists single-use CSRF `state` + PKCE verifier with TTL.
- `complete_oauth_login(provider: str, code: str, state: str) -> AuthResult`
  - Verifies CSRF `state`, exchanges auth code via `Authlib`, links existing user **only** if `email_verified=True`, creates `auth_identities` + `user_sessions`.
- `validate_session(session_token: str) -> Optional[AuthenticatedSession]`
  - Verifies HMAC signature & `SHA-256(token)` lookup in `user_sessions` (`revoked_at IS NULL`, `expires_at > now`).
- `logout(session_token: str) -> None` — Sets `user_sessions.revoked_at = now`.

### `UserService` (`qna_bot/services/user_service.py`)
- `get_profile(user_id: str) -> UserProfile`
- `update_profile(user_id: str, full_name: str, avatar_url: str = "") -> UserProfile`
- `change_password(user_id: str, current_password: str, new_password: str) -> None` — Re-verifies current password, updates hash, revokes other active sessions.
- `get_settings(user_id: str) -> Dict[str, Any]` / `save_settings(user_id: str, updates: Mapping[str, Any]) -> Dict[str, Any]`
- `delete_account(user_id: str) -> None` — Deletes/tombstones SQLite user rows, Qdrant vectors, local files, and active sessions.

---

## 2. Storage & Document Services (`qna_bot/storage/`, `qna_bot/rag/`, `qna_bot/services/`)

### `StorageProvider` / `LocalStorageProvider` (`qna_bot/storage/local.py`)
- `save_file(user_id: str, bucket: Literal["documents", "images", "attachments", "exports"], filename: str, data: bytes) -> StoredFileMeta`
  - Generates collision-safe path under `storage/users/{safe_user_id}/{bucket}/{uuid}_{safe_name}`, checks canonical `Path.resolve().is_relative_to(...)`, computes `SHA-256` checksum, and persists bytes atomically.
- `read_file(user_id: str, relative_path: str) -> bytes` — Enforces user-root prefix + traversal check.
- `delete_file(user_id: str, relative_path: str) -> bool`
- `delete_user_storage(user_id: str) -> int`

### `VectorStoreBackend` / `QdrantVectorStoreManager` (`qna_bot/rag/vector_store.py`)
Preserves existing backend-agnostic interface while upgrading implementation from FAISS to Qdrant:
- `has_document_hash(user_id: str, file_hash: str, workspace_id: str = "default") -> bool`
- `add_documents(user_id: str, docs: Sequence[Document], document_meta: Mapping[str, Any]) -> int`
  - Upserts deterministic point UUIDs with payload `{user_id, workspace_id, document_id, source, chunk_index, file_hash, created_at}`.
- `search_documents(user_id: str, query: str, k: int = 4, score_threshold: float = 0.2, workspace_id: str = "default") -> List[RetrievedChunk]`
  - Injects mandatory `Qdrant Filter` on `user_id` (+ optional `workspace_id`).
- `delete_document(user_id: str, source: str, document_id: Optional[str] = None) -> int`
- `list_documents(user_id: str) -> List[Dict[str, Any]]`
- `clear(user_id: str) -> None`

### `DocumentService` (`qna_bot/services/document_service.py`)
- `ingest_uploaded_file(user_id: str, file_bytes: bytes, filename: str, workspace_id: str = "default") -> IngestionResult`
  - Step 1: Validate type/size + compute checksum -> deduplicate per user.
  - Step 2: Save raw bytes via `LocalStorageProvider` + insert `uploaded_documents(status="processing")`.
  - Step 3: Extract/chunk/embed -> upsert Qdrant -> mark `"indexed"` (on Qdrant/embedding failure, retain file + mark `"failed"` with `error_message` for retry).
- `retry_indexing(user_id: str, document_id: str) -> IngestionResult`
- `delete_document(user_id: str, document_id: str) -> bool` — Deletes Qdrant points, SQLite row, and local file.

---

## 3. Conversation, Chat, MCP & Suggestion Services

### `ConversationService` (`qna_bot/services/conversation_service.py`)
- `create_thread(user_id: str, chat_mode: str = "Auto (Agentic)", title: str = "New Conversation", thread_id: Optional[str] = None) -> str`
- `list_threads_grouped(user_id: str, search_query: str = "", include_archived: bool = False) -> Dict[str, List[Dict[str, Any]]]`
- `rename_thread(thread_id: str, user_id: str, new_title: str) -> None`
- `set_pinned(thread_id: str, user_id: str, pinned: bool) -> None`
- `set_archived(thread_id: str, user_id: str, archived: bool) -> None`
- `delete_thread(thread_id: str, user_id: str) -> bool`
- `record_feedback(user_id: str, message_id: int, rating: Literal["up", "down"], comment: str = "") -> None`

### `ChatService` (`qna_bot/services/chat_service.py`)
- `run_agentic_turn(..., on_event: Optional[Callable[[Dict[str, Any]], None]] = None) -> ChatTurnResult`
- `resume_pending_approval(..., approval_id: str, approved: bool, ...) -> ChatTurnResult`

### `MCPService` (`qna_bot/services/mcp_service.py`)
- Wraps user-scoped registration, credential encryption (`CredentialVault`), OAuth2 PKCE flows (`OAuthFlowManager`), registry discovery (`MCPRegistryService`), per-tool permission toggles, health checks, and execution logs (`mcp_execution_logs`).

### `SuggestionService` (`qna_bot/services/suggestion_service.py`)
- `get_suggestions(user_id: str, current_chat_id: str = "", model_name: str = "", cache_store: Optional[MutableMapping[str, Any]] = None, force_refresh: bool = False, allow_llm: bool = False) -> List[SuggestedPrompt]`
  - Caches prompts per context signature (`mcp_tools + documents + recent_titles`) to prevent rerun flickering.
