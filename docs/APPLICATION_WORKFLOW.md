# QnAGPT Application Lifecycle & End-to-End Workflows

**Project**: QnAGPT – Production-Ready Multi-User Agentic AI Workspace  
**Status**: Verified Implementation  
**Author**: Principal Software Architect & Senior Python Backend Engineer  

---

## 1. Startup & Bootstrapping Workflow

```mermaid
sequenceDiagram
    participant OS as CLI / Streamlit Runner
    participant Init as qna_bot/__init__.py
    participant App as qna_bot/app.py
    participant Cfg as qna_bot/config/settings.py
    participant DB as qna_bot/database/session.py
    participant Mig as qna_bot/database/legacy_migration.py
    participant UI as Streamlit UI Session State

    OS->>Init: streamlit run qna_bot/app.py
    Init->>Init: sanitize_import_path() (ensures project root at sys.path[0])
    Init->>App: main()
    App->>Cfg: get_settings() (loads .env, st.secrets, defaults)
    App->>App: configure_logging(), configure_page()
    App->>UI: initialize_session_state()
    App->>App: apply_styles() (injects light-mode styles.css)
    App->>DB: _get_shared_orm_db(database_url)
    App->>Mig: ensure_legacy_workspace_synced(orm_db)
    Mig->>DB: PRAGMA table_info inspection & column sync
    App->>App: build_platform_services(settings)
```

1. **Path Sanitation**: `qna_bot/__init__.py` runs `sanitize_import_path()` to ensure the project root is at index 0 of `sys.path`.
2. **Configuration Hot-Reloading**: `get_settings()` inspects file modification signatures of `.env` and `.env.local`, merging active process variables and `st.secrets` into an immutable `AppSettings` dataclass.
3. **Database Connection Pool**: `_get_shared_orm_db()` caches a singleton `SQLAlchemyDatabase` instance with WAL mode and `PRAGMA busy_timeout = 10000`.
4. **Schema Synchronization**: `ensure_legacy_workspace_synced()` inspects table schemas dynamically to apply non-destructive backward-compatibility migrations (such as adding `is_pinned` to `threads` if absent).
5. **Platform Services Instantiation**: Factory `build_platform_services()` instantiates `LocalStorageProvider`, `QdrantVectorStoreManager`, `AuthService`, and `UserService`.

---

## 2. Authentication & Authorization Lifecycle

```mermaid
flowchart TD
    START([User Enters Application]) --> CHECK_TOKEN{Session Token in st.session_state?}

    CHECK_TOKEN -- "Token Present" --> VALIDATE_SESSION["AuthService.validate_session(token)"]
    VALIDATE_SESSION --> IS_VALID{Session Active & Not Revoked?}
    IS_VALID -- "Yes" --> RESOLVE_USER["Bind user_id to session state<br/>Build runtime services"]
    RESOLVE_USER --> DISPATCH_PAGE["Dispatch Workspace Router"]

    CHECK_TOKEN -- "No Token" --> RENDER_AUTH["Render Auth Gate (auth_views.py)"]
    IS_VALID -- "No (Expired/Revoked)" --> RENDER_AUTH

    RENDER_AUTH --> AUTH_CHOICE{Login Method Selected}

    %% Email / Password Flow
    AUTH_CHOICE -- "Email & Password" --> RATE_CHECK{Sliding Window Rate Limit?}
    RATE_CHECK -- "> 5 Failures in 300s" --> BLOCK["Block Login (RateLimitExceededError)"]
    RATE_CHECK -- "Permitted" --> VERIFY_PASS["Verify Argon2id Password Hash"]
    VERIFY_PASS -- "Mismatch" --> RECORD_FAIL["Record Failed Attempt<br/>Return 'Invalid email or password'"]
    VERIFY_PASS -- "Valid" --> CREATE_SESSION["Generate 32-byte URL-safe token<br/>Store HMAC-SHA256 digest in user_sessions"]
    CREATE_SESSION --> RESOLVE_USER

    %% OAuth2 Flow
    AUTH_CHOICE -- "Google / GitHub OAuth2" --> START_OAUTH["AuthService.start_oauth_login(provider)"]
    START_OAUTH --> PKCE["Generate SHA-256 PKCE Challenge & State"]
    PKCE --> REDIRECT["Redirect Browser to Identity Provider"]
    REDIRECT --> CALLBACK["Callback Received (?code=...&state=...)"]
    CALLBACK --> EXCHANGE["AuthService.complete_oauth_login(...)<br/>Exchange Code with Provider"]
    EXCHANGE --> LINK_USER["Lookup/Create User & AuthIdentity"]
    LINK_USER --> CREATE_SESSION
```

1. **Session Resolution**: Every request checks `st.session_state.auth_session_token`. The token is hashed with HMAC-SHA256 and matched against `user_sessions` where `revoked_at IS NULL` and `expires_at > now`.
2. **Rate Limiting**: `SlidingWindowRateLimiter` enforces a strict ceiling of $\le 5$ failed attempts per 300 seconds per client key.
3. **Argon2id Hashing**: Passwords are cryptographically salted and hashed using `argon2-cffi`. Plaintext passwords never enter log streams or database records.
4. **OAuth2 PKCE Handshake**: Complies with RFC 7636 Proof Key for Code Exchange (PKCE) with cryptographically random state tokens to prevent CSRF attacks.

---

## 3. Document Ingestion & Chunking Workflow

```mermaid
flowchart LR
    UPLOAD["Uploaded File (PDF, DOCX, CSV, TXT, MD)"] --> VALIDATE["DocumentService.validate_upload()<br/>(Check MIME, Size <= 20 MB)"]
    VALIDATE --> SAVE_DISK["LocalStorageProvider.save_file()<br/>storage/users/{user_id}/documents/"]
    SAVE_DISK --> EXTRACT["rag/loaders.py: load_document()<br/>(pypdf, docx2txt, csv.DictReader)"]
    EXTRACT --> CHUNK["rag/chunking.py: chunk_text()<br/>(Recursive Splitter: 1000/200)"]
    CHUNK --> EMBED["rag/embeddings.py<br/>(sentence-transformers all-MiniLM-L6-v2)"]
    EMBED --> STORE[("Qdrant Vector Store<br/>Payload: user_id, document_id, page")]
    STORE --> RECORD_DB[("SQLAlchemy Document Repository<br/>Status: 'indexed'")]
```

1. **Validation**: Files are checked against `settings.allowed_document_types` and `settings.max_document_size_mb`.
2. **Tenant Directory Isolation**: The original raw file is written to `storage/users/{user_id}/documents/{document_id}_{filename}` with path traversal safeguards.
3. **Multi-Format Extraction**: Text is extracted based on file extension (`pypdf` for PDF, `docx2txt` for DOCX, standard `csv` reader, UTF-8 text loader).
4. **Recursive Chunking**: Chunks are generated with 1,000 characters length and 200 characters overlap.
5. **Dense Vector Indexing**: Chunks are embedded locally using `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) and inserted into Qdrant with payload metadata filters (`user_id`, `document_id`, `source`, `page`).

---

## 4. Chat Request Turn & Corrective RAG (CRAG) Workflow

```mermaid
sequenceDiagram
    participant User as User / Browser
    participant Canvas as ui/chat_view.py & composer.py
    participant ChatSvc as services/chat_service.py
    participant Graph as agent/graph.py (CompiledStateGraph)
    participant Nodes as agent/nodes/ (Modular Handlers)
    participant CRAG as rag/crag/pipeline.py
    participant Qdrant as Qdrant Vector Engine
    participant DDG as DuckDuckGo Search
    participant Groq as Groq Cloud API

    User->>Canvas: Submits query in Chat Composer
    Canvas->>ChatSvc: stream_response(query, chat_mode, user_id)
    ChatSvc->>Graph: invoke() with initial AgentState
    Graph->>Nodes: analyze_query_node (intent classification)
    Nodes->>Groq: Intent classification prompt
    Groq-->>Nodes: query_type = "DOCUMENT"

    Graph->>Nodes: retrieve_documents_node
    Nodes->>Qdrant: Similarity search (filtered by user_id)
    Qdrant-->>Nodes: List of RetrievedChunk objects

    Graph->>Nodes: grade_documents
    Nodes->>CRAG: evaluate_and_assemble(query, documents)
    CRAG->>Groq: DocumentEvaluator multi-factor grading rubric
    Groq-->>CRAG: EvaluationResult (scores & confidence)

    alt Confidence >= 0.70 (CORRECT)
        CRAG->>Nodes: Filter to salient evidence sentences
    else 0.40 <= Confidence < 0.70 (AMBIGUOUS)
        CRAG->>Nodes: QueryRewriter resolves anaphora
        Nodes->>DDG: Live web search query
        DDG-->>Nodes: Web search results
        Nodes->>Groq: Cross-verify internal + web evidence
    else Confidence < 0.40 (INCORRECT)
        CRAG->>Nodes: Full web research fallback
        Nodes->>DDG: Search and fetch top-3 webpages
    end

    Nodes-->>Graph: EvidenceBundle assembled
    Graph->>Nodes: generate_answer (unbound Response LLM)
    Nodes->>Groq: Grounded answer prompt with citations
    Groq-->>Nodes: Generated answer text
    Graph->>Nodes: validate_answer (checks for hallucinations)
    Nodes-->>ChatSvc: Final validated answer & citations
    ChatSvc-->>Canvas: Streams tokens to UI
```

---

## 5. Autonomous Agent Workspace & HITL Execution Workflow

```mermaid
sequenceDiagram
    participant User as User (Streamlit UI)
    participant AgentSvc as services/agent_service.py
    participant Graph as agent/graph.py (CompiledStateGraph)
    participant MCP as mcp_client/client_manager.py
    participant Appr as services/approval_service.py
    participant Server as External MCP Server (Stdio / SSE)

    User->>AgentSvc: Submits complex goal in Agent Workspace
    AgentSvc->>Graph: invoke() in "Agent Workspace" mode
    Graph->>Graph: plan_task (breaks goal into sequential steps)
    loop For each planned step
        Graph->>Graph: select_tools (Agent LLM selects tool & arguments)
        Graph->>Graph: check_permissions (evaluates permission category)

        alt Tool Permission == "ALLOW_ALWAYS" (READ_ONLY)
            Graph->>MCP: execute_tool(namespaced_tool_name, arguments)
            MCP->>Server: Run tool over stdio/SSE
            Server-->>MCP: Tool result output
            MCP-->>Graph: ToolMessage with result
        else Tool Permission == "REQUIRE_APPROVAL" (MODIFICATION / SENSITIVE)
            Graph->>Appr: create_pending_approval(tool, arguments, reason)
            Graph->>Graph: interrupt(pending_approval)
            Graph-->>User: Halts graph; UI displays Approval Card

            User->>User: Reviews tool parameters & masked arguments
            User->>AgentSvc: Clicks "Approve Action" or "Reject Action"
            AgentSvc->>Appr: verify_and_consume(approval_id)
            AgentSvc->>Graph: Command(resume={"approved": true})

            alt Human Approved
                Graph->>MCP: execute_tool(tool, arguments)
                MCP->>Server: Run tool over stdio/SSE
                Server-->>MCP: Tool result output
                MCP-->>Graph: ToolMessage
            else Human Rejected
                Graph->>Graph: Append "Action rejected by user" ToolMessage
            end
        end

        Graph->>Graph: observe_results & evaluate_progress
    end

    Graph->>Graph: generate_response (synthesizes completed steps)
    Graph-->>AgentSvc: Final task response
    AgentSvc-->>User: Displays completed plan and execution timeline
```

---

## 6. Conversation Export Workflow

The application supports one-click serializations of conversation threads:
1. **JSON Export (`export_as_json`)**: Exports complete metadata, message roles, timestamps, tool calls, and model names in RFC 8259 JSON.
2. **Markdown Export (`export_as_markdown`)**: Exports a human-readable Markdown transcript formatted with blockquotes, headers, and bulleted citations.
3. **Plain Text Export (`export_as_text`)**: Clean text document suitable for copy-pasting or archival without formatting tags.
