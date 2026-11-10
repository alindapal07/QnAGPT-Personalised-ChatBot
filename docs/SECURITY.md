# QnAGPT Security Architecture & Controls

## 1. Threat Model & Trust Boundaries

1. **Client UI -> Application Services**: The Streamlit UI is treated strictly as an untrusted presentation layer. `user_id` is **never** trusted from query params, widget keys, or client dicts—protected backend services derive the authenticated `user_id` exclusively from a verified `UserSession` lookup (`token_hash`, `revoked_at IS NULL`, `expires_at > now`).
2. **External Web & MCP Endpoints**: User-supplied MCP URLs (`stdio`, `streamable_http`, `sse`) and web search URLs (`SafeWebPageReader`) are treated as potential SSRF, command-injection, or prompt-injection vectors.
3. **Uploaded User Files & Embeddings**: Uploaded files (`pdf`, `txt`, `docx`, `csv`, `md`) and user photos may contain oversized payloads, path traversal filenames (`../../.env`), or malicious macros.

---

## 2. Authentication, Passwords & Sessions

| Control | Implementation | Details |
|---|---|---|
| Password Hashing | `argon2-cffi` (`Argon2id`) | Automatic per-password salt, memory-hard hashing; plaintext passwords never stored or logged. |
| Password Policy | `AuthService` validator | Minimum `8` characters requiring uppercase, lowercase, and numeric/symbol diversity; normalized email (`strip().lower()`). |
| Login Rate Limiting | Sliding-window limiter | `max 5` failed password attempts per `5 minutes` per normalized email/client key; returns uniform `"Invalid email or password"` error (no user enumeration). |
| OAuth2 (Google / GitHub) | `Authlib` + CSRF `state` | Server-side single-use `state` + nonce verification (`10-min` TTL). Auto-links to existing password accounts **only** when provider returns `email_verified = true`. Zero fake OAuth bypasses when client credentials are unset. |
| Session Lifecycle | `user_sessions` table | Stores `SHA-256` hash of high-entropy token (`secrets.token_urlsafe(32)`), explicit `expires_at`, CSRF token, and server-side revocation (`revoked_at`) on logout or password change. |
| Account Deletion | Atomic teardown | Revokes all sessions, purges user Qdrant points (`user_id` filter), deletes `storage/users/{user_id}/`, and cascades/removes user database rows & vault secrets. |

---

## 3. Data, Storage & Vector Isolation

- **SQLite Relational Queries**: Every repository query on conversations, messages, documents, MCP servers, credentials, permissions, approvals, feedback, and settings requires `user_id == auth_user_id`. Direct mutations verify row ownership inside the SQLite transaction and fail closed.
- **Qdrant Vector Isolation**: Every point stores `user_id`, `workspace_id`, `document_id`, `source`, and `chunk_index` metadata payload. Both `query_points` and `delete` inject mandatory `models.Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id)), ...])`. Cross-tenant filters (`tenant_b` querying `tenant_a`) return zero hits.
- **Local Filesystem Storage (`LocalStorageProvider`)**:
  - Sanitizes `user_id` and filenames (`basename` + character whitelist + UUID prefix).
  - Enforces canonical `Path.resolve().is_relative_to(storage_root)` — raises `SecurityError` on any `..` or symlink escape.
  - Validates extension, size (`<= 20MB` docs / `<= 10MB` images), and `SHA-256` checksum; files are never placed on Python import path or executed.

---

## 4. Cryptography & Secret Management

- **At-Rest Vault (`CredentialVault`)**: Encrypts MCP bearer tokens, API keys, STDIO env secrets, and OAuth refresh/access tokens using Fernet (`AES-128-CBC` + `HMAC-SHA256`). Master key loaded from `QNAGPT_CREDENTIAL_KEY` env var (or persisted with `0600` permissions in `data/.mcp_vault.key`).
- **Log & UI Redaction (`mask_secrets`, `sanitize_tool_output`)**: Regex and key-name redaction strips keys containing `token`, `secret`, `api_key`, `password`, `authorization`, `cookie`, and bearer/Groq/GitHub/OpenAI token prefixes before writing logs or returning MCP results to the LLM/UI.
- **Parent Env Isolation**: STDIO MCP child processes never inherit parent secrets listed in `BLOCKED_PARENT_ENV_KEYS` (`GROQ_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `HF_TOKEN`, `AWS_*`, `GITHUB_TOKEN`).

---

## 5. Network SSRF, Prompt Injection & HITL Controls

- **SSRF Validation (`validate_remote_url` & `SafeWebPageReader`)**: Blocks non-HTTP(S) schemes, localhost/`.local` names, loopback (`127.0.0.0/8`), private RFC1918 ranges, link-local metadata endpoints (`169.254.169.254`), and DNS rebinding across HTTP redirects (`follow_redirects=False` with manual hop validation).
- **Prompt Injection Defense**: Retrieved RAG chunks, fetched web pages, and MCP tool outputs are wrapped cleanly as untrusted reference context (`ToolMessage` / quoted source blocks) for the unbound Response LLM (`_make_response_llm`), never promoted to system instructions.
- **Backend-Enforced HITL Approvals (`ApprovalManager`)**: `MODIFICATION` and `SENSITIVE` tools require a persisted `APPROVED` row matching `(user_id, thread_id, server_id, tool_name, arguments_hash)`. Even if an attacker replays an approval ID with modified arguments, `compute_arguments_hash` rejects execution cleanly, and single consumption sets `status = "CONSUMED"`.
