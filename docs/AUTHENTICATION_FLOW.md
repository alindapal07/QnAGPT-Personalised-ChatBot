# QnAGPT Authentication & Session Flow Specification

**Project**: QnAGPT – Production-Ready Multi-User Agentic AI Workspace  
**Status**: Verified Implementation  
**Author**: Principal Software Architect & Senior Python Backend Engineer  

---

## 1. Supported Identity Providers

1. **Email & Password**:
   - Normalized email formatting: `email.strip().lower()`.
   - Minimum password length: $\ge 8$ characters with lowercase, uppercase, and numeric/symbol diversity.
   - Cryptographic hashing via `argon2-cffi` (`Argon2id` algorithm with unique per-password cryptographic salt).
   - Constant-time hash verification preventing timing-attack vectors.
   - Generic authentication failure response (`"Invalid email or password"`) preventing user enumeration.
2. **Google Sign-In**:
   - OAuth 2.0 / OIDC using `Authlib` with `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET`.
   - Automatic account linking only when `email_verified == true`.
3. **GitHub Sign-In**:
   - OAuth 2.0 using `Authlib` with `GITHUB_CLIENT_ID` and `GITHUB_CLIENT_SECRET`.
   - Fetches verified primary email via GitHub API (`https://api.github.com/user/emails`).

*Note*: If OAuth credentials are not provided in `.env`, the UI renders a helpful notification banner and keeps standard Email/Password authentication fully functional.

---

## 2. Sliding-Window Rate Limiting

To prevent credential stuffing and brute-force attacks, the application implements `SlidingWindowRateLimiter` (`qna_bot/services/auth_service.py`):

```python
class SlidingWindowRateLimiter:
    def __init__(self, max_attempts: int = 5, window_seconds: int = 300):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
```

- **Policy**: Maximum of **5 failed attempts** within a **300-second (5-minute)** sliding window per client identifier (email or IP address).
- **Enforcement**: If the attempt count exceeds 5, `AuthService` raises `RateLimitExceededError` with an exact countdown of remaining retry seconds.
- **Reset**: Successful authentication immediately clears the failure record for that identifier.

---

## 3. Session Token Lifecycle

```mermaid
sequenceDiagram
    participant Client as User Browser / Streamlit
    participant Auth as AuthService (qna_bot/services/auth_service.py)
    participant Crypto as Security Core (qna_bot/core/security.py)
    participant DB as SQLite user_sessions Table

    Client->>Auth: authenticate_user(email, password)
    Auth->>Crypto: verify_password(password, stored_argon2id_hash)
    Crypto-->>Auth: True (Credentials valid)
    Auth->>Crypto: generate_secure_token(nbytes=32)
    Crypto-->>Auth: Raw 32-byte URL-safe bearer token (e.g., "sec_tok_...")
    Auth->>Crypto: hash_token(raw_token)
    Crypto-->>Auth: HMAC-SHA256 digest of bearer token
    Auth->>DB: INSERT INTO user_sessions (token_hash, user_id, expires_at)
    Auth-->>Client: Returns raw bearer token
    Client->>Client: Stores token in st.session_state.auth_session_token

    Note over Client,DB: Subsequent Requests
    Client->>Auth: validate_session(raw_token)
    Auth->>Crypto: hash_token(raw_token)
    Auth->>DB: SELECT * FROM user_sessions WHERE token_hash = ? AND revoked_at IS NULL
    DB-->>Auth: Session record (expires_at > now)
    Auth-->>Client: Authenticated UserSession object
```

### Security Properties:
- **No Plaintext Session Storage**: Only HMAC-SHA256 digests are persisted in SQLite. A compromised database snapshot cannot be used to hijack active user sessions.
- **Configurable TTL**: Sessions expire after `settings.session_ttl_hours` (default: 168 hours / 7 days).
- **Explicit Revocation**: Logging out marks `revoked_at = CURRENT_TIMESTAMP`, immediately invalidating the session digest across all devices.

---

## 4. OAuth2 Authorization Handshake with PKCE

Complies with RFC 7636 Proof Key for Code Exchange (PKCE) to prevent code interception attacks:

```mermaid
sequenceDiagram
    participant User as User Browser
    participant UI as Streamlit Auth Page
    participant Auth as AuthService
    participant OP as OAuth Provider (Google / GitHub)
    participant DB as SQLite Platform Database

    User->>UI: Clicks "Sign in with Google / GitHub"
    UI->>Auth: initiate_oauth_flow(provider)
    Auth->>Auth: Generate random 32-byte state & code_verifier
    Auth->>Auth: Compute SHA-256 code_challenge
    Auth-->>UI: Authorization URL with state & code_challenge
    UI-->>User: Redirects browser to Provider consent screen

    User->>OP: Grants permission
    OP-->>User: Redirects to APP_BASE_URL/?code=...&state=...
    User->>UI: Streamlit loads with query params

    UI->>Auth: complete_oauth_flow(provider, code, state)
    Auth->>Auth: Validate state parameter (CSRF protection)
    Auth->>OP: POST /token with code & code_verifier
    OP-->>Auth: Access token & ID token
    Auth->>OP: GET /userinfo (or /user/emails)
    OP-->>Auth: Verified profile & verified email

    Auth->>DB: Check if auth_identities exists
    alt Existing Linked Account
        Auth->>DB: Resolve existing user_id
    else New Account
        Auth->>DB: INSERT INTO users (email, is_verified=True)
        Auth->>DB: INSERT INTO auth_identities (user_id, provider, provider_user_id)
    end

    Auth->>DB: Create user_sessions record
    Auth-->>UI: Active bearer session token
    UI->>UI: Clears URL query params & reruns into Workspace
```
